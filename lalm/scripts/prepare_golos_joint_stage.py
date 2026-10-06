"""Prepare one finite Golos + unchanged joint-replay epoch; never launch training."""
import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples, DynamicBucketingSampler
from lhotse.dataset.sampling.base import TokenConstraint
from lalm_core.data_module import LALMDataset, estimate_cut_tokens
from lalm_core.model import LALMProcessor
from scripts.prepare_phone2_context_stage import identities
from scripts.prepare_projector_semantic_v3 import MODEL, ROOT, SEED, sha256

GOLOS = ROOT / 'golos-farfield-full-staging-v1'
EXIT = ROOT / 'golos-farfield-full-staging-v1-code/export.exit'
REPLAY = ROOT / 'runs/asr-projector-joint-selfteacher-v1'
REPLAY_SHA = '5da88bbd0ea6bf2e185fc8182f8b40a8c33a72ef99843180d3bd4651f9edc906'
OUTPUT = ROOT / 'runs/asr-projector-golos-joint-v1'
VALID = ROOT / 'short-context-stage4/validation.yaml'
VALID_SHA = 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc'
FROZEN = ['audio_tower', 'language_model', 'asr_head', 'asr_norm']
TEST = ROOT / 'golos-farfield-official-test-v1'
TEST_GATE = TEST / 'train-test-intersection-qa.json'


def check_full_qa(qa):
    assert qa['status'] == 'CPU_QA_PASS_STAGING_ONLY'
    for key in ('original_farfield_rows', 'streamed_source_rows',
                'unique_source_ids', 'original_mirror_text_matches'):
        assert qa[key] == 124003, key
    assert qa['full_original_id_set_equal']
    for key in ('missing_ids', 'extra_ids', 'official_test_id_overlap'):
        assert qa[key] == [], key
    assert qa['native_heldout_id_file_pcm_intersections'] == 0
    assert qa['native_processor_batch16_labels_eos_pass']
    assert qa['all_native_texts_roles_paths_checked']
    assert qa['unique_native_pcm_hours'] >= 100


def check_test_overlap(qa, train_manifest_sha, train_index_sha):
    """Final launch gate: all test audio, including the blank-reference row."""
    assert qa['status'] == 'PASS'
    assert qa['train_manifest_sha256'] == train_manifest_sha
    assert qa['train_source_index_sha256'] == train_index_sha
    assert qa['train_source_rows'] == 124003 and qa['test_decoded_rows'] == 1916
    assert qa['test_native_cuts'] == 1915 and qa['test_unresolved_rows'] == 1
    assert qa['intersections'] == dict(source_ids=0, encoded_sha256=0, pcm_sha256=0)
    assert qa['test_source_index_path'] == str(TEST / 'source-index.jsonl')
    return {str(TEST / 'source-index.jsonl'): qa['test_source_index_sha256'],
            str(TEST / 'eval-asr.jsonl.gz'): qa['test_manifest_sha256']}


def audit_sampler(cuts, golos_ids, target_hours=100):
    """Native two-rank order, no model/audio loading; count unique PCM separately."""
    samplers = []
    for rank in (0, 1):
        native = cuts.resample(16000).repeat(1).map(lambda c: estimate_cut_tokens(c, 12.5))
        sampler = DynamicBucketingSampler(native, constraint=TokenConstraint(max_tokens=2000),
            shuffle=True, num_buckets=5, buffer_size=10000, shuffle_buffer_size=25000,
            drop_last=False, world_size=2, rank=rank, seed=0)
        sampler.set_epoch(0)
        samplers.append(sampler)
    emitted, seen_golos, pcm_seconds = set(), set(), {}
    rank_stats = [Counter(), Counter()]
    order_hashes = [hashlib.sha256(), hashlib.sha256()]
    milestones, first_target, unique_seconds = [], None, 0.0
    batches = 0
    for batches, pair in enumerate(zip_longest(*samplers), 1):
        assert all(batch is not None for batch in pair), 'Unequal DDP batch counts'
        for rank, batch in enumerate(pair):
            for cut in batch:
                order_hashes[rank].update((cut.id + '\n').encode())
                emitted.add(cut.id)
                original_id = cut.custom.get('original_id')
                source = 'golos' if original_id in golos_ids else 'replay'
                task = cut.custom.get('task', 'answer')
                rank_stats[rank][source + '_cuts'] += 1
                rank_stats[rank][task + '_cuts'] += 1
                rank_stats[rank][source + '_seconds'] += cut.duration
                if source == 'golos':
                    seen_golos.add(original_id)
                    pcm = cut.custom['pcm_sha256']
                    if pcm not in pcm_seconds:
                        pcm_seconds[pcm] = cut.duration
                        unique_seconds += cut.duration
        hours = unique_seconds / 3600
        if first_target is None and hours >= target_hours:
            first_target = batches
        if batches % 500 == 0:
            milestones.append(dict(updates=batches, unique_golos_pcm_hours=hours,
                rank_exposure=[dict(c) for c in rank_stats]))
    # Native repeat(1) uses its own view IDs; include any DDP tail repetitions only
    # in exposure counts, never in unique source or audio-hour coverage.
    expected = set(cuts.resample(16000).repeat(1).ids)
    assert emitted == expected, (len(expected - emitted), len(emitted - expected))
    assert seen_golos == golos_ids
    assert first_target is not None, 'Unique decoded Golos audio did not reach target hours'
    return dict(batches_per_rank={'0': batches, '1': batches}, updates=batches,
        num_steps=batches - 1, N100=first_target, target_unique_hours=target_hours,
        unique_golos_source_ids=len(seen_golos), unique_golos_pcm=len(pcm_seconds),
        unique_golos_pcm_hours=sum(pcm_seconds.values()) / 3600,
        full_native_view_id_coverage=True, rank_exposure=[dict(c) for c in rank_stats],
        ordered_rank_id_sha256=[h.hexdigest() for h in order_hashes], milestones=milestones,
        world_size=2, grad_accum_steps=1, epoch=0, sampler_seed=0, drop_last=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output_dir
    assert out.resolve().is_relative_to(ROOT / 'runs') and not out.exists(), out
    assert EXIT.read_text().strip() == '0'
    qa_path, summary_path = GOLOS / 'full-cpu-qa.json', GOLOS / 'summary.json'
    qa, summary = json.loads(qa_path.read_text()), json.loads(summary_path.read_text())
    check_full_qa(qa)
    assert summary['source_dataset'] == 'Sh1man/golos_opus'
    assert summary['revision'] == '931f5c412a8045e0b03b9e5584072f0810c41dcc'
    assert summary['split'] == 'train' and summary['config'] == 'farfield'
    assert summary['counts']['source_rows'] == 124003 and summary['label_text_unchanged']
    ready = json.loads((REPLAY / 'readiness.json').read_text())
    assert ready['manifest_sha256'] == REPLAY_SHA and ready['cuts'] == 21588
    assert ready['tasks'] == {'asr': 19068, 'answer': 2520}
    assert ready['model_dir'] == str(MODEL) and ready['trainable_parameters'] == 10490880
    assert ready['frozen_modules'] == FROZEN
    assert len(ready['trainable_tensors']) == 4
    assert sum(math.prod(s) for s in ready['trainable_tensors'].values()) == 10490880
    golos_path = GOLOS / 'pilot-asr.jsonl.gz'
    checks = {str(EXIT): sha256(EXIT), str(qa_path): sha256(qa_path),
        str(summary_path): sha256(summary_path), str(REPLAY / 'readiness.json'): sha256(REPLAY / 'readiness.json'),
        str(golos_path): qa['manifest_sha256'], str(REPLAY / 'train.jsonl.gz'): REPLAY_SHA,
        str(VALID): VALID_SHA, **summary['input_sha256'], **ready['model_file_sha256'],
        **ready['heldout_manifest_sha256']}
    checks.update({str(GOLOS / name): digest for name, digest in summary['output_sha256'].items()})
    assert qa['manifest_sha256'] == summary['output_sha256']['pilot-asr.jsonl.gz']
    for path, digest in checks.items():
        assert sha256(path) == digest, path
    golos, replay = [CutSet.from_file(p).to_eager() for p in (golos_path, REPLAY / 'train.jsonl.gz')]
    assert len(golos) == qa['native_cuts'] == summary['counts']['native_staging_cuts']
    assert len(golos) > 0
    assert len(replay) == 21588
    replay_rows = {c.id: hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest() for c in replay}
    golos_ids = {c.custom['original_id'] for c in golos}
    assert len(golos_ids) == len(golos)
    heldout = identities([c for p in ready['heldout_manifest_sha256'] for c in CutSet.from_file(p)])
    overlap = {k: len(v & heldout[k]) for k, v in identities(golos).items()}
    assert not any(overlap.values()), overlap
    processor = LALMProcessor.from_pretrained(MODEL)
    totals, sample = Counter(), []
    for name, source in (('golos', golos), ('replay', replay)):
        for cut in source:
            task = cut.custom.get('task', 'answer')
            assert task in ('asr', 'answer') and .5 <= cut.duration <= 30
            target = cut.supervisions[0].custom['answer']
            assert cut.conversation[-1] == {'role': 'assistant', 'content': target}
            assert cut.rendered_conversation.endswith(target + '<|im_end|>\n')
            if task == 'asr':
                assert target == cut.supervisions[0].text
            if name == 'golos':
                assert task == 'asr' and cut.custom['split'] == 'train'
                assert not cut.custom['unresolved_reasons'] and not cut.custom.get('history')
            target_n = len(processor.tokenizer.encode(target + '<|im_end|>', add_special_tokens=False))
            text_n = len(processor.tokenizer.encode(cut.rendered_conversation.strip(), add_special_tokens=False))
            assert text_n == cut.num_text_tokens, cut.id
            assert target_n <= 256 and text_n + round(cut.duration * 12.5) <= 2000, cut.id
            totals[name + '_' + task + '_target_tokens'] += target_n
        for task, count in (('asr', 6), ('answer', 4 if name == 'replay' else 0)):
            ordered = sorted((c for c in source if c.custom.get('task', 'answer') == task), key=lambda c:(c.duration,c.id))
            sample.extend(ordered[round(i * (len(ordered)-1)/(count-1))] for i in range(count))
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[CutSet.from_cuts(sample)]
    assert batch['batch_size'] == 16 and batch['asr_mask'].sum().item() == 12
    for cut, labels, mask in zip(batch['cuts'], batch['labels'], batch['asr_mask']):
        assert mask.item() == (cut.custom.get('task', 'answer') == 'asr')
        assert processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False).strip() == cut.supervisions[0].custom['answer'] + '<|im_end|>', cut.id
    # Existing native finite mux, each source exactly once. Neither source's
    # labels/history/custom metadata is changed; no OASST or admission placeholder.
    cuts = CutSet.mux(golos, replay, weights=[len(golos), len(replay)], seed=SEED, stop_early=False).to_eager()
    assert len(cuts) == len(set(cuts.ids)) == len(golos) + 21588
    out.mkdir()
    manifest = out / 'train.jsonl.gz'
    cuts.to_file(manifest)
    saved_replay = {c.id: hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()
        for c in CutSet.from_file(manifest) if c.id in replay_rows}
    assert saved_replay == replay_rows, 'Replay rows changed during composition'
    # Reload: estimate_cut_tokens adds metadata; never let this mutate saved rows.
    sampler = audit_sampler(CutSet.from_file(manifest), golos_ids)
    assert abs(sampler['unique_golos_pcm_hours'] - qa['unique_native_pcm_hours']) < 1e-6
    (out / 'train.yaml').write_text(yaml.safe_dump([dict(name='golos-joint', manifest=str(manifest),
        hours=sum(c.duration for c in cuts)/3600, weights=1)]))
    (out / 'validation.yaml').write_bytes(VALID.read_bytes())
    report = dict(status='prepared_requires_parent_review_not_launched', manifest=str(manifest),
        manifest_sha256=sha256(manifest), train_config_sha256=sha256(out / 'train.yaml'),
        validation_config_sha256=sha256(out / 'validation.yaml'), input_sha256=checks,
        model_dir=str(MODEL), cuts=len(cuts), tasks=dict(Counter(c.custom.get('task','answer') for c in cuts)),
        golos_cuts=len(golos), replay_cuts=len(replay), replay_manifest_sha256=REPLAY_SHA,
        unchanged_replay_rows_verified=len(saved_replay),
        frozen_modules=FROZEN, trainable_parameters=10490880, trainable_tensors=ready['trainable_tensors'],
        finite_sampler=sampler, target_tokens=dict(totals), native_cpu_batch=dict(cuts=16, asr=12, answer=4,
            exact_final_target_and_EOS=True, ids=[c.id for c in batch['cuts']]), heldout_overlap=overlap,
        preparation_script_sha256=sha256(__file__), training_launched=False,
        official_test_overlap=dict(status='pending_final_launch_gate', audit=str(TEST_GATE),
            note='Launch requires actual1916 decoded test rows, including1 blank reference; preparation does not claim PASS.'),
        notes=['No source text/history rewriting, OASST, synthetic answer generation or trainer changes.',
            'Source staging metadata retained; preparation alone is not training admission.',
            'Fresh optimizer from iter26000. start_batch does not restore sampler position.',
            'At update500, export only after post-save Saving trainer checkpoint log; hardlink protects keep_last_k=2 deletion at1500. CPU export/GPU1 eval must not restart training sampler.',
            'Evaluate raw weights, all fixed rows and character loops. No automatic deployment or proved speaker-separation claim.'])
    (out / 'readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'updates': sampler['updates'], 'N100': sampler['N100']}), flush=True)


if __name__ == '__main__':
    main()
