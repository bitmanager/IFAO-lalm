"""Bounded ASR-readout diagnostic: native CutSet composition and CPU preflight only."""
import hashlib
import json
import struct
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMProcessor
from scripts.prepare_phone2_context_stage import identities

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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUTPUT.exists(), OUTPUT
    assert Path('/runs/ifao-projector-calls-stage.exit').read_text().strip() == '0'
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, path
    new_ready = json.loads(NEW.with_name('readiness.json').read_text())
    assert new_ready['status'] == 'ready' and new_ready['cuts'] == 29947
    assert new_ready['manifest_sha256'] == EXPECTED[NEW]
    # The v2 hf/ directory contains config only: require the completed HF export.
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    model_files = sorted(p for p in MODEL.iterdir() if p.is_file())
    model_hashes = {str(p): sha256(p) for p in model_files}
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

    full = CutSet.from_file(BASE).to_eager()
    assert len(full) == 40464
    sources = {
        'v2-asr-only': full.filter(lambda c: c.custom.get('task') == 'asr').to_eager(),
        'rnnt-shard00': CutSet.from_file(NEW).to_eager(),
    }
    assert len(sources['v2-asr-only']) == 26226
    assert len(sources['rnnt-shard00']) == 29947
    valid_specs = yaml.safe_load(VALID.read_text())
    heldout_paths = {x['manifest'] for x in valid_specs}
    # Retain the broader frozen phone/SVQ/history checks from shard preparation.
    heldout_paths.update(new_ready['heldout_QA']['manifests'])
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
        sample.extend(ordered[round(i * (len(ordered)-1) / 7)] for i in range(8))
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
                              training_recipe='asr-readout-diagnostic-v1')
        preflight[name] = dict(cuts=len(cuts), hours=sum(c.duration for c in cuts)/3600,
                               heldout_overlap=overlap,
                               ids_sha256=hashlib.sha256('\n'.join(sorted(identity['cut_ids'])).encode()).hexdigest())
    cuts = CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()],
                     seed=20261006, stop_early=False).to_eager()
    assert len(cuts) == len({c.id for c in cuts}) == 56173
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
    config_path.write_text(yaml.safe_dump([dict(name='asr-readout-diagnostic-v1',
        manifest=str(manifest), hours=hours, weights=1)]))
    report = dict(status='ready_for_review_not_launched', cuts=len(cuts), tasks={'asr': len(cuts)},
        task_exposure_hours=hours, manifest=str(manifest), manifest_sha256=sha256(manifest),
        train_config_sha256=sha256(config_path), input_sha256={str(p): h for p,h in EXPECTED.items()},
        model_dir=str(MODEL), model_file_sha256=model_hashes,
        frozen_modules=['audio_tower', 'language_model', 'projector'],
        frozen_semantic_components=['GigaAM', 'audio projector', 'Qwen backbone and embeddings', 'Qwen agent readout'],
        trainable_tensors=shapes, trainable_parameters=388277760,
        source_preflight=preflight, heldout_manifest_sha256={p:sha256(p) for p in sorted(heldout_paths)},
        source_qa_reused=str(NEW.with_name('readiness.json')),
        native_cpu_batch=dict(cuts=16, asr=16, exact_final_target_and_EOS=True),
        labels_history_system_unchanged=True, all_cut_target_EOS_checked=len(cuts),
        training_launched=False, main_training_connected=False,
        notes=['ASR readout repair diagnostic only: no projector/encoder/Qwen training.',
               'Automatic RNNT labels remain pseudo labels, not gold.',
               'Fixed validation unchanged; full native ASR evaluation is a separate job.',
               'Agent generation is structurally independent of the readout; post-export frozen-state hashes and matched deterministic agent smoke are required before claiming bit-identical outputs.'])
    (OUTPUT / 'readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
