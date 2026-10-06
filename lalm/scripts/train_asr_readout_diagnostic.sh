#!/usr/bin/env bash
# Invoke only after review. Stock IFAO/Auden, fresh optimizer, head-only 500 updates.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-asr-readout-diagnostic.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-readout-diagnostic-v1
export IFAO_MODEL_PATH=/runs/dev-storage/ifao-data/runs/asr-projector-calls-v2/export/epoch-2
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG=/runs/dev-storage/ifao-data/short-context-stage4/validation.yaml
test "$(cat /runs/ifao-projector-calls-stage.exit)" = 0
cd /ifao-context-review/lalm
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

out = Path(os.environ['IFAO_RUN_DIR'])
ready = json.loads((out / 'readiness.json').read_text())
assert ready['status'] == 'ready_for_review_not_launched' and ready['cuts'] == 56173
assert ready['tasks'] == {'asr': 56173}
assert ready['trainable_parameters'] == 388277760
assert ready['native_cpu_batch']['exact_final_target_and_EOS']
assert ready['model_dir'] == os.environ['IFAO_MODEL_PATH']
assert not (out/'hf').exists(), 'Fresh run only; do not reuse optimizer/checkpoints'
checks = {**ready['input_sha256'], **ready['model_file_sha256'], **ready['heldout_manifest_sha256'],
          ready['manifest']: ready['manifest_sha256'],
          os.environ['IFAO_TRAIN_CONFIG']: ready['train_config_sha256']}
for path, expected in checks.items(): assert sha(path) == expected, path
print('Preflight passed: 56173 ASR views; only asr_norm.weight/asr_head.weight expected trainable (388.277760M).', flush=True)
PY
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=1 trainer.start_batch=0 \
  trainer.num_epochs=1 trainer.num_steps=499 trainer.grad_accum_steps=1 \
  'trainer.frozen_modules=[audio_tower,language_model,projector]' \
  trainer.optimizer.lr=1e-6 trainer.optimizer.weight_decay=0.0 trainer.scheduler.eta_min=1e-6 \
  trainer.valid_interval=500 trainer.save_every_n=1 trainer.keep_last_k=2
