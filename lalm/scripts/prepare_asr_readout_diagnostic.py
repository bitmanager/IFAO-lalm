"""Bounded ASR-readout diagnostic: native CutSet composition and CPU preflight only."""
import argparse
import hashlib
import inspect
import json
import random
import struct
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMProcessor
from scripts.prepare_phone2_context_stage import identities
from scripts.prepare_projector_semantic_v3 import finite_sampler_audit

ROOT = Path('/runs/dev-storage/ifao-data')
OUTPUT = ROOT / 'runs/asr-readout-diagnostic-v1'
MODEL = ROOT / 'runs/asr-projector-calls-v2/export/epoch-2'
BASE = ROOT / 'runs/asr-projector-calls-v2/train.jsonl.gz'
NEW = ROOT / 'phone1-remaining-rnnt-v2/shard-00/native/train.jsonl.gz'
VALID = ROOT / 'short-context-stage4/validation.yaml'
EXPECTED = {
    BASE: 'b6a058dd63c4cde165c16a1e0539bb1b2184892db2814e577f2298d2c6effe2a',
    NEW: '94ad883827fd81a8f0ba1bca0ff266c592901b2fb163a28f618f88197f4b0713',
    VALID: 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc',
}


def load_recipe(path=None):
    if path is not None:
        recipe = json.loads(Path(path).read_text())
        assert recipe['updates'] == 1000 and recipe['lr'] == 1e-5
        assert recipe['expected_cuts'] == 205634
        assert len(recipe['sources']) == 8
        assert recipe['model_file_sha256'], 'Immutable initialization hashes required'
        return recipe
    return dict(name='asr-readout-diagnostic-v1', output=str(OUTPUT), model=str(MODEL),
        validation=str(VALID), validation_sha256=EXPECTED[VALID], updates=500, lr=1e-6,
        valid_interval=500, expected_cuts=56173, completion_marker='/runs/ifao-projector-calls-stage.exit',
        exit_marker='/runs/ifao-asr-readout-diagnostic.exit', sources=[
            dict(name='v2-asr-only', manifest=str(BASE), sha256=EXPECTED[BASE],
                 input_cuts=40464, task_filter='asr', cuts=26226),
            dict(name='rnnt-shard00', manifest=str(NEW), sha256=EXPECTED[NEW],
                 input_cuts=29947, cuts=29947, readiness=str(NEW.with_name('readiness.json')))])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, help='One source/hash/budget specification; omit for original v1')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    OUTPUT, MODEL, VALID = (Path(recipe[k]) for k in ('output', 'model', 'validation'))
    EXPECTED = {Path(s['manifest']): s['sha256'] for s in recipe['sources']}
    EXPECTED[VALID] = recipe['validation_sha256']
    assert not OUTPUT.exists(), OUTPUT
    if recipe.get('completion_marker'):
        assert Path(recipe['completion_marker']).read_text().strip() == '0'
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, path
    source_readiness = {}
    for spec in recipe['sources']:
        if spec.get('readiness'):
            path = Path(spec['readiness'])
            if spec.get('readiness_sha256'):
                assert sha256(path) == spec['readiness_sha256']
            ready = json.loads(path.read_text())
            assert ready['status'] == 'ready' and ready['cuts'] == spec['cuts']
            assert ready['manifest_sha256'] == spec['sha256']
            assert not any(ready['heldout_QA']['comparisons'].values())
            source_readiness[str(path)] = ready
            EXPECTED[path] = sha256(path)
    # The v2 hf/ directory contains config only: require the completed HF export.
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    model_files = sorted(p for p in MODEL.iterdir() if p.is_file())
    model_hashes = {str(p): sha256(p) for p in model_files}
    if recipe.get('model_file_sha256'):
        assert model_hashes == recipe['model_file_sha256']
    assert all((MODEL / name).is_file() for name in set(index['weight_map'].values()))
    config = json.loads((MODEL / 'config.json').read_text())
    assert config['asr_layer'] == 35
    frozen = ('audio_tower.', 'language_model.', 'projector.')
    readout_keys = {k for k in index['weight_map'] if not k.startswith(frozen)}
    assert readout_keys == {'asr_norm.weight', 'asr_head.weight'}, readout_keys
    shapes = {}
    for key in sorted(readout_keys):
        with open(MODEL / index['weight_map'][key], 'rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            shapes[key] = json.loads(stream.read(length))[key]['shape']
    assert shapes == {'asr_head.weight': [151670, 2560], 'asr_norm.weight': [2560]}

    sources = {}
    for spec in recipe['sources']:
        cuts = CutSet.from_file(spec['manifest']).to_eager()
        assert len(cuts) == spec['input_cuts']
        if spec.get('task_filter'):
            cuts = cuts.filter(lambda c: c.custom.get('task') == spec['task_filter']).to_eager()
        if spec.get('take'):
            cuts = cuts.shuffle(rng=random.Random(spec['selection_seed'])).subset(first=spec['take'])
        assert len(cuts) == spec['cuts']
        if spec.get('selected_ids_sha256'):
            assert hashlib.sha256('\n'.join(sorted(c.id for c in cuts)).encode()).hexdigest() == spec['selected_ids_sha256']
        assert spec['name'] not in sources
        sources[spec['name']] = cuts
    valid_specs = yaml.safe_load(VALID.read_text())
    heldout_paths = {x['manifest'] for x in valid_specs}
    # Retain the broader frozen phone/SVQ/history checks from shard preparation.
    for ready in source_readiness.values():
        for path, digest in ready['heldout_QA']['manifests'].items():
            assert sha256(path) == digest, path
            heldout_paths.add(path)
    heldout = [c for path in sorted(heldout_paths) for c in CutSet.from_file(path)]
    held_ids = identities(heldout)
    preflight = {}
    seen = set()
    sample = []
    for name, cuts in sources.items():
        identity = identities(cuts)
        overlap = {key: len(value & held_ids[key]) for key, value in identity.items()}
        assert not any(overlap.values()), (name, overlap)
        assert not seen & identity['cut_ids']
        seen.update(identity['cut_ids'])
        ordered = sorted(cuts, key=lambda c: (c.duration, c.id))
        n = 16 // len(sources)
        sample.extend(ordered[round(i * (len(ordered)-1) / (n-1))] for i in range(n))
        for cut in cuts:
            assert cut.custom.get('task') == 'asr'
            assert cut.custom.get('split', 'train') == 'train'
            assert .5 <= cut.duration <= 30 and len(cut.supervisions) == 1
            sup = cut.supervisions[0]
            assert sup.text and sup.text.strip() and sup.custom['answer'] == sup.text
            assert cut.custom['conversation'][-1] == {'role': 'assistant', 'content': sup.text}
            assert cut.custom['rendered_conversation'].endswith(sup.text + '<|im_end|>\n')
            # Only experiment admission metadata changes; source files remain untouched.
            cut.custom.update(training_eligible=True, main_training_connected=False,
                              training_recipe=recipe['name'])
        preflight[name] = dict(cuts=len(cuts), hours=sum(c.duration for c in cuts)/3600,
                               heldout_overlap=overlap,
                               ids_sha256=hashlib.sha256('\n'.join(sorted(identity['cut_ids'])).encode()).hexdigest())
    cuts = CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()],
                     seed=20261006, stop_early=False).to_eager()
    assert len(cuts) == len({c.id for c in cuts}) == recipe['expected_cuts']
    sampler_audit = finite_sampler_audit(cuts, required_updates=recipe['updates'])
    processor = LALMProcessor.from_pretrained(MODEL)
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[CutSet.from_cuts(sample)]
    assert batch['asr_mask'].all() and batch['batch_size'] == 16
    for cut, labels in zip(batch['cuts'], batch['labels']):
        target = processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
        assert target.strip() == cut.supervisions[0].text + '<|im_end|>', cut.id
    OUTPUT.mkdir()
    manifest = OUTPUT / 'train.jsonl.gz'
    cuts.to_file(manifest)
    hours = sum(c.duration for c in cuts) / 3600
    config_path = OUTPUT / 'train.yaml'
    config_path.write_text(yaml.safe_dump([dict(name=recipe['name'],
        manifest=str(manifest), hours=hours, weights=1)]))
    report = dict(status='ready_for_review_not_launched', cuts=len(cuts), tasks={'asr': len(cuts)},
        task_exposure_hours=hours, manifest=str(manifest), manifest_sha256=sha256(manifest),
        train_config_sha256=sha256(config_path), input_sha256={str(p): h for p,h in EXPECTED.items()},
        model_dir=str(MODEL), model_file_sha256=model_hashes,
        frozen_modules=['audio_tower', 'language_model', 'projector'],
        frozen_semantic_components=['GigaAM', 'audio projector', 'Qwen backbone and embeddings', 'Qwen agent readout'],
        trainable_tensors=shapes, trainable_parameters=388277760,
        source_preflight=preflight, heldout_manifest_sha256={p:sha256(p) for p in sorted(heldout_paths)},
        source_qa_reused=list(source_readiness),
        native_cpu_batch=dict(cuts=16, asr=16, exact_final_target_and_EOS=True),
        labels_history_system_unchanged=True, all_cut_target_EOS_checked=len(cuts),
        training_launched=False, main_training_connected=False, recipe=recipe,
        recipe_sha256=sha256(args.recipe) if args.recipe else None,
        finite_sampler=sampler_audit, updates=recipe['updates'], lr=recipe['lr'],
        code_sha256={str(p): sha256(p) for p in (
            Path(__file__).resolve(), Path(__file__).with_name('train_asr_readout_diagnostic.sh').resolve(),
            Path(inspect.getfile(finite_sampler_audit)), Path(inspect.getfile(identities)))},
        notes=['ASR readout repair diagnostic only: no projector/encoder/Qwen training.',
               'Automatic RNNT labels remain pseudo labels, not gold.',
               'Fixed validation unchanged; full native ASR evaluation is a separate job.',
               'Agent generation is structurally independent of the readout; post-export frozen-state hashes and matched deterministic agent smoke are required before claiming bit-identical outputs.'])
    (OUTPUT / 'readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
