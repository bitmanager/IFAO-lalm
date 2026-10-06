"""One-run data recipe: use native Lhotse selection/replay/mux, no new loader."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from lhotse import CutSet


root = Path('/runs/dev-storage/ifao-data')
output = root / 'short-context-stage4'
sova_path = root / 'sova-asr-v1/prepared/train-audited.jsonl.gz'
target_path = root / 'context-target-v1/train.jsonl.gz'
sova_summary = json.loads((sova_path.parent / 'summary.json').read_text())
sova = CutSet.from_file(sova_path).to_eager()
target = CutSet.from_file(target_path).to_eager()
assert len(sova) == sova_summary['counts']['kept'] and len(target) == 1162
assert hashlib.sha256(sova_path.read_bytes()).hexdigest() == sova_summary['train_manifest_sha256']
assert all(c.custom['split'] == 'train' and c.clean_source_qc['clean_qc_candidate'] for c in target)
target_qc = json.loads((target_path.parent / 'readiness.json').read_text())
assert target_qc['source_pcm_split_overlap'] == 0 and target_qc['labels_history_system_preserved']
assert target_qc['all_audio_decoded'] and target_qc['source_hashes_matched_pre_inference_clean_qc']
assert hashlib.sha256(target_path.read_bytes()).hexdigest() == target_qc['manifest_sha256'][target_path.name]
for cut in target:
    cut.custom.update(training_eligible=True, main_training_connected=True,
                      training_recipe='short-context-stage4')
output.mkdir(exist_ok=False)

sources = {
    'sova': sova,
    'phone1_replay': CutSet.from_file(root / 'openstt-phone1/train-asr.jsonl.gz').to_eager()
        .shuffle(rng=random.Random(20261006)).subset(first=12000),
    'youtube_replay': CutSet.from_file(root / 'youtube-exp-v1/prepared/train-asr.jsonl.gz').to_eager()
        .shuffle(rng=random.Random(20261006)).subset(first=4000),
    'balalaika_replay': CutSet.from_file('/ifao-context-data/asr-balalaika-v1/train.jsonl.gz').to_eager()
        .shuffle(rng=random.Random(20261006)).subset(first=6000),
    'context_replay': CutSet.from_file('/ifao-context-data/asr-v1/train-joint.jsonl.gz').repeat(4),
    'context_target_replay': target.repeat(4),
}
summary = {'sources': {}, 'recipe': 'native finite CutSet.mux(stop_early=False)',
           'notes': ['Repeat counts are exposure, not new audio.',
                     'ASR and answer views share recordings.',
                     'Phone/YouTube labels are automatic; teacher answers are not factual gold.',
                     'Context-target source admission: 83 training pairs with exact clean ASR on both roles.',
                     'Fixed 12-pair validation and all SVQ examples are excluded.']}
for name, cuts in sources.items():
    cuts = cuts.to_eager()
    sources[name] = cuts
    bins = Counter('lt2' if c.duration < 2 else '2to5' if c.duration < 5 else 'ge5' for c in cuts)
    summary['sources'][name] = {'cuts': len(cuts), 'task_exposure_hours': sum(c.duration for c in cuts) / 3600,
                                'duration_bins': dict(bins)}

manifest = output / 'train.jsonl.gz'
CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()],
           seed=20261006, stop_early=False).to_file(manifest)
cuts = CutSet.from_file(manifest)
ids = set()
tasks = Counter()
for cut in cuts:
    assert cut.id not in ids, cut.id
    ids.add(cut.id)
    tasks[cut.custom.get('task', 'answer')] += 1
assert len(ids) == sum(len(c) for c in sources.values())
hours = sum(s['task_exposure_hours'] for s in summary['sources'].values())
summary.update(cuts=len(ids), tasks=dict(tasks), task_exposure_hours=hours,
               manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
               input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (sova_path, target_path)})
(output / 'train.yaml').write_text(yaml.safe_dump([
    {'name': 'short-speech-context-replay', 'manifest': str(manifest), 'hours': hours, 'weights': 1}]))
validation = yaml.safe_load(Path('/ifao-context-data/asr-balalaika-v1/validation.yaml').read_text())
validation.append({'name': 'context-target-fixed12',
                   'manifest': str(root / 'context-target-v1/validation.jsonl.gz'),
                   'hours': 0.28005555555555556, 'weights': 1})
(output / 'validation.yaml').write_text(yaml.safe_dump(validation))
(output / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
