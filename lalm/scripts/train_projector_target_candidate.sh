#!/usr/bin/env bash
# Separate bounded candidate: native data composition and projector-only tuning.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-projector-target-candidate.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
root=/runs/dev-storage/ifao-data
export IFAO_RUN_DIR="$root/runs/asr-projector-target-candidate-v1"
export IFAO_MODEL_PATH="$root/runs/asr-short-context-v1/export/iter-29000"
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$root/short-context-stage4/validation.yaml"
python - <<'PY'
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import yaml
from lhotse import CutSet

root = Path('/runs/dev-storage/ifao-data')
output = Path(os.environ['IFAO_RUN_DIR'])
checks = {
    root / 'context-target-v1/train.jsonl.gz': 'aef8c9a5811033c9fbfd0ad70f24e1946a97558a22ebe0b661ec5fcb53c3a1de',
    root / 'context-target-reserve29-v1/train.jsonl.gz': '752d05fddb1ba19a2eef201dc828f7814128883ae90f82f538890ae6056f45f8',
    root / 'context-target-older441-v1/train.jsonl.gz': '11ef22aa9c599dd367ebde9c4290bb8fbc4c6405e891ea327183475b240efe49',
    root / 'short-context-stage4/validation.yaml': 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc',
}
for path, digest in checks.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
sources = {}
for path in list(checks)[:3]:
    cuts = CutSet.from_file(path).to_eager()
    for cut in cuts:
        assert cut.custom['split'] == 'train'
        cut.custom.update(training_eligible=True, main_training_connected=False,
                          training_recipe='projector-target-candidate-v1')
    sources[path.parent.name] = cuts
sources['phone1-replay'] = CutSet.from_file(root / 'openstt-phone1/train-asr.jsonl.gz').to_eager().shuffle(rng=random.Random(20261006)).subset(first=6000)
sources['clean-context-replay'] = CutSet.from_file('/ifao-context-data/asr-v1/train-joint.jsonl.gz').to_eager()
cuts = CutSet.mux(*sources.values(), weights=[len(c) for c in sources.values()],
                 seed=20261006, stop_early=False).to_eager()
assert len(cuts) == sum(len(c) for c in sources.values()) == 18852
assert len({c.id for c in cuts}) == len(cuts)
assert all(0.5 <= c.duration <= 30 for c in cuts)
validation = yaml.safe_load(Path(os.environ['IFAO_VALID_CONFIG']).read_text())
heldout_ids = {c.id for spec in validation for c in CutSet.from_file(spec['manifest'])}
assert not heldout_ids.intersection(c.id for c in cuts)
output.mkdir(exist_ok=False)
manifest = output / 'train.jsonl.gz'
cuts.to_file(manifest)
hours = sum(c.duration for c in cuts) / 3600
Path(os.environ['IFAO_TRAIN_CONFIG']).write_text(yaml.safe_dump([
    {'name': 'target-and-clean-replay', 'manifest': str(manifest), 'hours': hours, 'weights': 1}]))
summary = {
    'cuts': len(cuts), 'tasks': dict(Counter(c.custom.get('task', 'answer') for c in cuts)),
    'task_exposure_hours': hours, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'sources': {name: {'cuts': len(cs), 'task_exposure_hours': sum(c.duration for c in cs) / 3600} for name, cs in sources.items()},
    'notes': ['Separate candidate; main training and validation unchanged.',
              'Task views and replay are not new unique audio.',
              'Bounded 1000-update experiment, not a complete data epoch.',
              'Fresh optimizer required for changed trainable parameter set.'],
}
(output / 'data-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
PY
cd /ifao-context-review/lalm
# Fresh optimizer; raw resume cannot load optimizer groups after freezing the head.
# Equal min/base LR uses the native cosine scheduler as a constant 1e-5 schedule.
CUDA_VISIBLE_DEVICES=1 python -u train.py --config-name gigaam_context \
  trainer.start_epoch=1 trainer.start_batch=0 trainer.num_epochs=1 trainer.num_steps=999 \
  'trainer.frozen_modules=[audio_tower,language_model,asr_head,asr_norm]' \
  trainer.optimizer.lr=1e-5 trainer.scheduler.eta_min=1e-5 \
  trainer.valid_interval=50000 trainer.save_every_n=1 trainer.keep_last_k=2
