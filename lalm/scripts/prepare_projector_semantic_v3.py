"""Bounded v3 operational recipe: existing native cuts, no generation or training."""
import hashlib
import json
import math
import random
import struct
from collections import Counter
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples, DynamicBucketingSampler
from lhotse.dataset.sampling.base import TokenConstraint
from lalm_core.data_module import LALMDataset, estimate_cut_tokens
from lalm_core.model import LALMProcessor
from scripts.prepare_phone2_context_stage import identities

ROOT = Path('/runs/dev-storage/ifao-data')
OUTPUT = ROOT / 'runs/asr-projector-semantic-v3'
MODEL = ROOT / 'runs/asr-short-context-v1/export/iter-26000'
VALID = ROOT / 'short-context-stage4/validation.yaml'
SEED = 20261006
PATHS = {
    'paired_asr': ROOT / 'context-history-flip-train-v1/train.jsonl.gz',
    'clean_asr': Path('/ifao-context-data/asr-v1/train-asr.jsonl.gz'),
    'original_answer': ROOT / 'original-script-answers-v1/train-answer.jsonl.gz',
    'phone_rnnt': ROOT / 'phone1-replay-rnnt-v2/train.jsonl.gz',
    'sova': ROOT / 'sova-asr-v1/prepared/train-audited.jsonl.gz',
}
EXPECTED = dict(zip(PATHS, [
    'fbcf10603fa9af4b0fbb6d4c27c1f7c2089aca43cce9978232b3752e986e5c2f',
    'da88be7567c972d27ec6dd7125da82f181b8fad985341ac450e3286ff0639631',
    '58b8a543a53b9b015aa4a4d139f11ddde8e96c2e5573b5d67c398b14ba11b863',
    '8c8f20222a344e20cbce52cd35dffd7e83b95812abba7c55e8aacaea80469f50',
    '98ed55ea04c2638d89341e0888a5d285a3e5f64352d56f6e35fe5faba0a13d20',
]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_sampler_audit(cuts):
    """Same native data/sampler settings, explicit DDP ranks, no audio/model pass."""
    counts = {}
    for rank in (0, 1):
        native = cuts.resample(16000).repeat(1).map(lambda c: estimate_cut_tokens(c, 12.5))
        sampler = DynamicBucketingSampler(native, constraint=TokenConstraint(max_tokens=2000),
            shuffle=True, num_buckets=5, buffer_size=10000, shuffle_buffer_size=25000,
            drop_last=False, world_size=2, rank=rank, seed=0)
        sampler.set_epoch(0)
        counts[str(rank)] = sum(1 for _ in sampler)
    assert min(counts.values()) >= 1500, counts
    return dict(batches_per_rank=counts, required_updates=1500, grad_accum_steps=1,
                native_sampler_seed=0, epoch=0, drop_last=False, world_size=2)


def main():
    assert not OUTPUT.exists(), OUTPUT
    for name, path in PATHS.items():
        assert sha256(path) == EXPECTED[name], path
    assert sha256(VALID) == 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc'
    sources = {name: CutSet.from_file(path).to_eager() for name, path in PATHS.items()}
    answers = {c.id: c for c in sources['original_answer']}
    assert len(answers) == 846
    sources['clean_asr'] = sources['clean_asr'].filter(lambda c: c.id.removesuffix('-asr') in answers).to_eager()
    assert len(sources['clean_asr']) == 846, 'Stop: matching existing clean ASR required'
    for c in sources['clean_asr']:
        original = answers[c.id.removesuffix('-asr')]
        assert c.recording.to_dict() == original.recording.to_dict()
        assert (c.start, c.duration, c.channel, c.supervisions[0].text) == (original.start, original.duration, original.channel, original.supervisions[0].text)
        assert all(c.custom[k] == original.custom[k] for k in ('history', 'system', 'split', 'source_group_id'))
    sources['sova'] = sources['sova'].shuffle(rng=random.Random(SEED)).subset(first=6000)
    assert len(sources['paired_asr']) == 6240 and len(sources['phone_rnnt']) == 5988
    phone_ready = json.loads(PATHS['phone_rnnt'].with_name('readiness.json').read_text())
    flip_ready = json.loads(PATHS['paired_asr'].with_name('readiness.json').read_text())
    held_hashes = dict(phone_ready['heldout_QA']['manifests'])
    for spec in yaml.safe_load(VALID.read_text()):
        held_hashes.setdefault(spec['manifest'], sha256(spec['manifest']))
    for path, digest in held_hashes.items():
        assert sha256(path) == digest, path
    heldout = [c for path in held_hashes for c in CutSet.from_file(path)]
    held_ids = identities(heldout)
    # Preserve the upstream full-stem/PCM and source grouping exclusion proof.
    for path, digest in flip_ready['source_and_heldout_hashes_before_and_after'].items():
        assert sha256(path) == digest, path
    processor = LALMProcessor.from_pretrained(MODEL)
    audit, seen, sample = {}, set(), []
    for name, cuts in sources.items():
        ids = identities(cuts)
        overlap = {k: len(v & held_ids[k]) for k, v in ids.items()}
        assert not any(overlap.values()), (name, overlap)
        assert not seen & ids['cut_ids'], name
        seen.update(ids['cut_ids'])
        token_totals, target_lengths, total_lengths = Counter(), [], []
        for c in cuts:
            task = c.custom.get('task', 'answer')
            assert task == ('answer' if name == 'original_answer' else 'asr')
            assert c.custom.get('split', 'train') == 'train' and .5 <= c.duration <= 30
            assert all(s.type == 'file' and Path(s.source).is_file() for s in c.recording.sources)
            target = c.supervisions[0].custom['answer']
            assert c.conversation[-1] == {'role': 'assistant', 'content': target}
            assert c.rendered_conversation.endswith(target + '<|im_end|>\n')
            if task == 'asr':
                assert target == c.supervisions[0].text
            else:
                assert c.answer_provenance['quality'] == 'original_script_synthetic_not_gold'
                assert not c.custom.get('source_call_id')
            target_n = len(processor.tokenizer.encode(target + '<|im_end|>', add_special_tokens=False))
            text_n = len(processor.tokenizer.encode(c.rendered_conversation.strip(), add_special_tokens=False))
            assert text_n == c.num_text_tokens
            audio_n = round(c.duration * 12.5)
            target_lengths.append(target_n); total_lengths.append(text_n + audio_n)
            token_totals.update(target_including_eos=target_n, text=text_n, estimated_audio=audio_n)
            c.custom.update(training_eligible=True, main_training_connected=False, training_recipe='projector-semantic-v3')
        ordered = sorted(cuts, key=lambda c: (c.duration, c.id))
        n = 4 if name == 'original_answer' else 3
        sample.extend(ordered[round(i * (len(ordered)-1)/(n-1))] for i in range(n))
        audit[name] = dict(base_cuts=len(cuts), task=task, base_exposure_hours=sum(c.duration for c in cuts)/3600,
            heldout_overlap=overlap, selected_ids_sha256=hashlib.sha256('\n'.join(sorted(ids['cut_ids'])).encode()).hexdigest(),
            base_token_totals=dict(token_totals), max_target_tokens_including_eos=max(target_lengths),
            max_estimated_total_tokens=max(total_lengths), over256_target_tokens=sum(n > 256 for n in target_lengths),
            over2000_total_tokens=sum(n > 2000 for n in total_lengths), over8192_total_tokens=sum(n > 8192 for n in total_lengths))
    # Three exposures of the same original answer; native repeat changes only view IDs.
    sources['original_answer'] = sources['original_answer'].repeat(3, preserve_id=False).to_eager()
    cuts = CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()], seed=SEED, stop_early=False).to_eager()
    assert len(cuts) == len({c.id for c in cuts}) == 21612
    tasks = Counter(c.custom.get('task', 'answer') for c in cuts)
    assert tasks == {'asr': 19074, 'answer': 2538}
    sampler_audit = finite_sampler_audit(cuts)
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[CutSet.from_cuts(sample)]
    assert batch['batch_size'] == 16 and batch['asr_mask'].sum().item() == 12
    for c, labels in zip(batch['cuts'], batch['labels']):
        assert processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False).strip() == c.supervisions[0].custom['answer'] + '<|im_end|>', c.id
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    shapes = {}
    for key, filename in index.items():
        if key.startswith('projector.'):
            with (MODEL / filename).open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                shapes[key] = json.loads(stream.read(length))[key]['shape']
    assert len(shapes) == 4 and sum(math.prod(s) for s in shapes.values()) == 10490880
    print('Data/CPU preflight passed; hashing immutable initialization.', flush=True)
    model_hashes = {str(p): sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    OUTPUT.mkdir()
    manifest = OUTPUT / 'train.jsonl.gz'; cuts.to_file(manifest)
    hours = sum(c.duration for c in cuts)/3600
    (OUTPUT / 'train.yaml').write_text(yaml.safe_dump([dict(name='projector-semantic-v3', manifest=str(manifest), hours=hours, weights=1)]))
    (OUTPUT / 'validation.yaml').write_bytes(VALID.read_bytes())
    for name in audit:
        repetition = 3 if name == 'original_answer' else 1
        audit[name].update(repetitions=repetition, final_views=len(sources[name]),
            exposure_token_totals={k:v*repetition for k,v in audit[name]['base_token_totals'].items()})
    report = dict(status='ready_for_review_not_launched', cuts=len(cuts), tasks=dict(tasks), task_exposure_hours=hours,
        seed=SEED, recipe='native finite mux(stop_early=False), original answers repeat(3, preserve_id=False)',
        manifest=str(manifest), manifest_sha256=sha256(manifest), train_config_sha256=sha256(OUTPUT/'train.yaml'),
        validation_config_sha256=sha256(OUTPUT/'validation.yaml'), heldout_manifest_sha256=held_hashes,
        input_sha256={str(PATHS[k]):v for k,v in EXPECTED.items()}, source_preflight=audit,
        model_dir=str(MODEL), model_file_sha256=model_hashes, trainable_tensors=shapes, trainable_parameters=10490880,
        frozen_modules=['audio_tower', 'language_model', 'asr_head', 'asr_norm'],
        native_cpu_batch=dict(cuts=16, asr=12, answer=4, exact_final_target_and_EOS=True, ids=[c.id for c in batch['cuts']]),
        source_audio_paths_all_exist=True, duplicate_cut_ids=0, training_launched=False,
        finite_sampler=sampler_audit,
        optimizer='fresh AdamW; native cosine with equal base/min LR 3e-6', updates=1500,
        notes=['Repeat/task views and derived mixtures are exposures, not new audio hours.',
               'Clean ASR and original answers deliberately share all 846 source utterances; paired mixtures also reuse admitted source turns.',
               'Original synthetic script answers are not human gold; phone RNNT labels are automatic pseudo labels.',
               'No call-answer or previous regenerated synthetic answer targets are present.',
               'Fixed validation unchanged; upstream full-stem exclusion proof reused, no speaker-disjoint claim.'])
    (OUTPUT/'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
