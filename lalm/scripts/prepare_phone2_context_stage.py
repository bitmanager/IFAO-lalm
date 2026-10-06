"""One-run Stage5 data recipe; native finite Lhotse operations, no trainer launch."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from lhotse import CutSet


SEED = 20261006
ROOT = Path('/runs/dev-storage/ifao-data')
OUTPUT = ROOT / 'phone2-context-stage5'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration_bin(duration):
    return ('<0.5' if duration < .5 else '0.5-1' if duration < 1 else
            '1-2' if duration < 2 else '2-5' if duration <= 5 else '>5')


def identities(cuts):
    return {
        'cut_ids': {c.id for c in cuts},
        'recording_ids': {c.recording.id for c in cuts},
        'audio_paths': {s.source for c in cuts for s in c.recording.sources},
        'source_groups': {c.custom['source_group_id'] for c in cuts
                          if c.custom.get('source_group_id')},
        'available_encoded_sha256': {c.custom['source']['audio_sha256'] for c in cuts
                                    if c.custom.get('source', {}).get('audio_sha256')},
    }


def source_pcm(cuts):
    return {c.clean_source_qc[f'{role}_pcm_float32_le_sha256'] for c in cuts
            for role in ('foreground', 'background')}


def main():
    paths = {
        'phone2_short': ROOT / 'openstt-phone2/prepared-short/train-asr.jsonl.gz',
        'phone2_remaining': ROOT / 'openstt-phone2/prepared-remaining/train-asr.jsonl.gz',
        'sova_replay': ROOT / 'sova-asr-v1/prepared/train-audited.jsonl.gz',
        'balalaika_replay': Path('/ifao-context-data/asr-balalaika-v1/train.jsonl.gz'),
        'youtube_replay': ROOT / 'youtube-exp-v1/prepared/train-asr.jsonl.gz',
        'context_replay': Path('/ifao-context-data/asr-v1/train-joint.jsonl.gz'),
        'context_target_replay': ROOT / 'context-target-v1/train.jsonl.gz',
    }
    assert not OUTPUT.exists(), OUTPUT
    input_hashes = {str(p): sha256(p) for p in paths.values()}
    sources = {name: CutSet.from_file(p).to_eager() for name, p in paths.items()}
    for name, expected in [('phone2_short', 155535), ('phone2_remaining', 399678),
                           ('sova_replay', 78628)]:
        ready = json.loads((paths[name].parent / 'readiness.json').read_text())
        assert ready['status'] == 'ready' and ready['ready_as_asr_batch']
        assert len(sources[name]) == ready['cuts'] == expected
        assert input_hashes[str(paths[name])] == ready['train_manifest_sha256']

    target = sources['context_target_replay']
    target_ready = json.loads((paths['context_target_replay'].parent / 'readiness.json').read_text())
    assert len(target) == 1162 and target_ready['train_pairs'] == 83
    assert target_ready['source_pcm_split_overlap'] == 0
    assert target_ready['all_audio_decoded'] and target_ready['labels_history_system_preserved']
    assert target_ready['source_hashes_matched_pre_inference_clean_qc']
    assert input_hashes[str(paths['context_target_replay'])] == target_ready['manifest_sha256']['train.jsonl.gz']
    assert len({c.clean_source_qc['pair_id'] for c in target}) == 83
    assert all(c.custom['split'] == 'train' and c.clean_source_qc['clean_qc_candidate']
               and c.clean_source_qc['clean_asr_exact_both'] for c in target)

    validation_path = ROOT / 'short-context-stage4/validation.yaml'
    validation_bytes = validation_path.read_bytes()
    validation_config = yaml.safe_load(validation_bytes)
    validation_sets = [CutSet.from_file(entry['manifest']).to_eager() for entry in validation_config]
    heldout = {key: set().union(*(identities(cuts)[key] for cuts in validation_sets))
               for key in identities([])}
    target_validation = CutSet.from_file(ROOT / 'context-target-v1/validation.jsonl.gz').to_eager()
    assert not source_pcm(target) & source_pcm(target_validation)

    for name, count in [('sova_replay', 12000), ('balalaika_replay', 6000), ('youtube_replay', 4000)]:
        sources[name] = sources[name].shuffle(rng=random.Random(SEED)).subset(first=count)
        assert len(sources[name]) == count
    assert len(sources['context_replay']) == 5110

    audit, identity_sets = {}, {}
    for name, cuts in sources.items():
        ids = identities(cuts)
        assert len(ids['cut_ids']) == len(cuts), name
        overlap = {key: len(value & heldout[key]) for key, value in ids.items()}
        assert not any(overlap.values()), (name, overlap)
        assert all(c.custom.get('split', 'train') == 'train' for c in cuts), name
        identity_sets[name] = ids
        audit[name] = {'base_cuts': len(cuts), 'heldout_overlap': overlap,
                       'available_encoded_hashes': len(ids['available_encoded_sha256']),
                       'selected_ids_sha256': hashlib.sha256('\n'.join(c.id for c in cuts).encode()).hexdigest()}
    cross_source = {}
    for i, name in enumerate(sources):
        for other in list(sources)[i + 1:]:
            overlap = {key: len(value & identity_sets[other][key])
                       for key, value in identity_sets[name].items()}
            # These task views intentionally reuse contextual training groups.
            # Cut IDs, stored audio and heldout membership must still be disjoint.
            allowed_groups = {name, other} == {'context_replay', 'context_target_replay'}
            assert not any(count for key, count in overlap.items()
                           if key != 'source_groups' or not allowed_groups), (name, other, overlap)
            cross_source[f'{name}/{other}'] = overlap
    # Mixtures intentionally derive from existing contextual training turns.
    # Record that relationship separately from duplicate cuts/files.
    context_ids = {c.id.removesuffix('-asr') for c in sources['context_replay']}
    target_source_ids = {c.clean_source_qc[f'{role}_source_cut_id'] for c in target
                         for role in ('foreground', 'background')}
    derived_context_overlap = len(context_ids & target_source_ids)
    print(json.dumps({'preflight': 'passed', 'base_sources': audit,
                      'derived_target_source_ids_also_in_context': derived_context_overlap}), flush=True)

    for c in target:
        c.custom.update(training_eligible=True, main_training_connected=False,
                        training_recipe='phone2-context-stage5')
    for name in ('context_replay', 'context_target_replay'):
        sources[name] = sources[name].repeat(8).to_eager()

    summary = {'seed': SEED, 'recipe': 'native finite CutSet.mux(stop_early=False)',
               'input_sha256': input_hashes, 'source_preflight': audit,
               'cross_source_overlap': cross_source,
               'target_train_pairs': 83, 'target_train_validation_pcm_overlap': 0,
               'derived_target_source_ids_also_in_context': derived_context_overlap,
               'validation_config_source': str(validation_path),
               'validation_config_sha256': hashlib.sha256(validation_bytes).hexdigest(),
               'validation_manifest_sha256': {entry['manifest']: sha256(entry['manifest'])
                                              for entry in validation_config},
               'sources': {}, 'training_launched': False,
               'notes': ['Task repetitions and ASR/answer views are exposures, not new audio hours.',
                         'Target mixtures reuse training source turns; no acoustic-uniqueness claim.',
                         'Phone2/YouTube labels are automatic ASR; SOVA annotations follow its source card.',
                         'Teacher answers are not factual gold; no new labels or history are created.',
                         'Validation is copied byte-for-byte from Stage4; no new heldout introduced.']}
    for name, cuts in sources.items():
        bins, seconds, tasks = Counter(), Counter(), Counter()
        for c in cuts:
            bucket = duration_bin(c.duration)
            bins[bucket] += 1
            seconds[bucket] += c.duration
            tasks[c.custom.get('task', 'answer')] += 1
        summary['sources'][name] = {'cuts': len(cuts), 'tasks': dict(tasks),
            'task_exposure_hours': sum(seconds.values()) / 3600,
            'duration_bins': {b: {'count': bins[b], 'hours': seconds[b] / 3600} for b in bins}}

    OUTPUT.mkdir(exist_ok=False)
    manifest = OUTPUT / 'train.jsonl.gz'
    CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()],
               seed=SEED, stop_early=False).to_file(manifest)
    ids, tasks = set(), Counter()
    for c in CutSet.from_file(manifest):
        assert c.id not in ids, c.id
        ids.add(c.id)
        tasks[c.custom.get('task', 'answer')] += 1
    assert len(ids) == sum(len(c) for c in sources.values()) == 627389
    hours = sum(s['task_exposure_hours'] for s in summary['sources'].values())
    summary.update(status='ready', cuts=len(ids), duplicate_cut_ids=0, tasks=dict(tasks),
                   task_exposure_hours=hours, manifest_sha256=sha256(manifest))
    (OUTPUT / 'train.yaml').write_text(yaml.safe_dump([
        {'name': 'phone2-context-stage5', 'manifest': str(manifest), 'hours': hours, 'weights': 1}]))
    (OUTPUT / 'validation.yaml').write_bytes(validation_bytes)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
