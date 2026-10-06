#!/usr/bin/env bash
# Bounded continuation using native Lhotse composition and Auden DDP resume.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-projector-calls-stage.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
root=/runs/dev-storage/ifao-data
previous="$root/runs/asr-projector-target-candidate-v1"
export IFAO_RUN_DIR="$root/runs/asr-projector-calls-v2"
export IFAO_MODEL_PATH="$previous/hf"
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$root/short-context-stage4/validation.yaml"
test "$(cat /runs/ifao-projector-target-candidate.exit)" = 0
test -s "$previous/epoch-1.pt"
cd /ifao-context-review/lalm
python - <<'PY'
import hashlib
import json
import os
import random
from pathlib import Path

import yaml
from lhotse import CutSet
from scripts.prepare_phone2_context_stage import identities

root = Path('/runs/dev-storage/ifao-data')
output = Path(os.environ['IFAO_RUN_DIR'])
previous = root / 'runs/asr-projector-target-candidate-v1'
phone = root / 'phone1-replay-rnnt-v2'
calls = root / 'context-calls-overlap-v1'
checks = {
    calls / 'train.jsonl.gz': 'b9a466d00c56a3ac29df4c9f4a7d3803aac6b7e659b2a083992a76d7d7800ba1',
    root / 'sova-asr-v1/prepared/train-audited.jsonl.gz': '98ed55ea04c2638d89341e0888a5d285a3e5f64352d56f6e35fe5faba0a13d20',
    Path(os.environ['IFAO_VALID_CONFIG']): 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc',
}
phone_ready = json.loads((phone / 'readiness.json').read_text())
assert phone_ready['status'] == 'ready'
checks[phone / 'train.jsonl.gz'] = phone_ready['manifest_sha256']
checks[previous / 'train.jsonl.gz'] = json.loads((previous / 'data-summary.json').read_text())['manifest_sha256']
for path, expected in checks.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
assert json.loads((calls / 'readiness.json').read_text())['native_cpu_batch']['labels_exact_final_answer_only']
old_phone = CutSet.from_file(root / 'openstt-phone1/train-asr.jsonl.gz').to_eager().shuffle(rng=random.Random(20261006)).subset(first=6000)
old_ids = {c.id for c in old_phone}
old = CutSet.from_file(previous / 'train.jsonl.gz').to_eager()
assert len(old_ids & {c.id for c in old}) == 6000
sources = {
    'existing-context': old.filter(lambda c: c.id not in old_ids).to_eager(),
    'phone-rnnt-pseudo-v2': CutSet.from_file(phone / 'train.jsonl.gz').to_eager(),
    'calls-context': CutSet.from_file(calls / 'train.jsonl.gz').to_eager(),
    'sova-replay': CutSet.from_file(root / 'sova-asr-v1/prepared/train-audited.jsonl.gz').to_eager().shuffle(rng=random.Random(20261006)).subset(first=6000),
}
assert len(sources['existing-context']) == 12852
assert len(sources['phone-rnnt-pseudo-v2']) == phone_ready['cuts']
assert {c.id for c in sources['phone-rnnt-pseudo-v2']} <= old_ids
assert len(sources['calls-context']) == 15624
validation = yaml.safe_load(Path(os.environ['IFAO_VALID_CONFIG']).read_text())
heldout = [c for spec in validation for c in CutSet.from_file(spec['manifest'])]
heldout_ids = identities(heldout)
for name, cs in sources.items():
    assert all(.5 <= c.duration <= 30 for c in cs), name
    assert not any(value & heldout_ids[key] for key, value in identities(cs).items()), name
    for c in cs:
        c.custom.update(training_eligible=True, training_recipe='projector-calls-v2')
cuts = CutSet.mux(*sources.values(), weights=[len(cs) for cs in sources.values()], seed=20261006, stop_early=False).to_eager()
assert len(cuts) == len({c.id for c in cuts}) == sum(len(cs) for cs in sources.values())
output.mkdir(exist_ok=False)
manifest = output / 'train.jsonl.gz'
cuts.to_file(manifest)
hours = sum(c.duration for c in cuts) / 3600
Path(os.environ['IFAO_TRAIN_CONFIG']).write_text(yaml.safe_dump([{'name': 'projector-calls-v2', 'manifest': str(manifest), 'hours': hours, 'weights': 1}]))
summary = {'cuts': len(cuts), 'task_exposure_hours': hours,
           'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'input_sha256': {str(p): h for p, h in checks.items()},
           'sources': {name: len(cs) for name, cs in sources.items()},
           'notes': ['Automatic phone labels replaced by our RNNT pseudo labels, not gold.',
                     'Task views, replay and derived mixtures are not new unique hours.',
                     'Only projector trained; bounded continuation, not a complete data epoch.',
                     'More data, changed targets and world size: not a one-factor ablation.']}
(output / 'data-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
PY
cp -a "$previous/hf" "$IFAO_RUN_DIR/hf"
ln -s "$previous/epoch-1.pt" "$IFAO_RUN_DIR/epoch-1.pt"
# Resume the same four projector tensors and native optimizer/scheduler state.
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=2 trainer.start_batch=0 \
  trainer.num_epochs=2 trainer.num_steps=2999 \
  'trainer.frozen_modules=[audio_tower,language_model,asr_head,asr_norm]' \
  trainer.optimizer.lr=1e-5 trainer.scheduler.eta_min=1e-5 \
  trainer.valid_interval=1000 trainer.save_every_n=1 trainer.keep_last_k=2
