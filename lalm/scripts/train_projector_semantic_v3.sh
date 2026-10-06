#!/usr/bin/env bash
# Review-gated invocation only. Fresh iter26000 initialization, no resume state.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-projector-semantic-v3.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-projector-semantic-v3
export IFAO_MODEL_PATH=/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$IFAO_RUN_DIR/validation.yaml"
cd /ifao-context-review/lalm
python - <<'PY'
import json, os
from pathlib import Path
from scripts.prepare_asr_readout_diagnostic import sha256
out = Path(os.environ['IFAO_RUN_DIR'])
r = json.loads((out/'readiness.json').read_text())
assert r['status'] == 'ready_for_review_not_launched' and r['cuts'] == 21612
assert r['tasks'] == {'asr': 19074, 'answer': 2538}
assert r['trainable_parameters'] == 10490880
assert r['native_cpu_batch']['exact_final_target_and_EOS']
assert min(r['finite_sampler']['batches_per_rank'].values()) >= 1500
assert r['model_dir'] == os.environ['IFAO_MODEL_PATH']
assert not (out/'hf').exists() and not list(out.glob('*.pt')), 'Fresh optimizer/run required'
checks = {**r['input_sha256'], **r['model_file_sha256'], **r['heldout_manifest_sha256'],
          r['manifest']: r['manifest_sha256'], os.environ['IFAO_TRAIN_CONFIG']: r['train_config_sha256'],
          os.environ['IFAO_VALID_CONFIG']: r['validation_config_sha256']}
for path, digest in checks.items(): assert sha256(path) == digest, path
print('Preflight passed: fresh iter26000, projector only, 1500 updates at constant 3e-6.', flush=True)
PY
# Native stop is global_step > num_steps; final validation/save runs after stop.
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=1 trainer.start_batch=0 \
  trainer.num_epochs=1 trainer.num_steps=1499 trainer.grad_accum_steps=1 \
  'trainer.frozen_modules=[audio_tower,language_model,asr_head,asr_norm]' \
  trainer.optimizer.lr=3e-6 trainer.optimizer.weight_decay=0.0 trainer.scheduler.eta_min=3e-6 \
  trainer.mixed_precision=bf16 trainer.valid_interval=1000 trainer.save_every_n=1 trainer.keep_last_k=2 \
  data.sampler.max_tokens=2000
