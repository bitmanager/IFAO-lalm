#!/usr/bin/env bash
# Invoke only after review. Stock IFAO/Auden, fresh optimizer, bounded head-only run.
set -euo pipefail
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd /ifao-context-review/lalm
if (( $# )); then
  test "$#" = 2 && test "$1" = --recipe
  export IFAO_READOUT_RECIPE="$2"
else
  export IFAO_READOUT_RECIPE=''
fi
configuration=$(PYTHONPATH="$script_dir:/ifao-context-review/lalm:$PYTHONPATH" python - <<'PY'
import os
from prepare_asr_readout_diagnostic import load_recipe
r = load_recipe(os.environ['IFAO_READOUT_RECIPE'] or None)
for value in (r['output'], r['model'], r['validation'], r['updates'], r['lr'],
              r['valid_interval'], r['exit_marker']):
    assert '\n' not in str(value)
    print(value)
PY
)
mapfile -t settings <<< "$configuration"
test "${#settings[@]}" = 7
export IFAO_RUN_DIR="${settings[0]}"
export IFAO_MODEL_PATH="${settings[1]}"
export IFAO_VALID_CONFIG="${settings[2]}"
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
updates="${settings[3]}"
learning_rate="${settings[4]}"
valid_interval="${settings[5]}"
exit_marker="${settings[6]}"
trap 'status=$?; printf "%s\n" "$status" > "$exit_marker"' EXIT
PYTHONPATH="$script_dir:/ifao-context-review/lalm:$PYTHONPATH" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from prepare_asr_readout_diagnostic import load_recipe

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

out = Path(os.environ['IFAO_RUN_DIR'])
recipe_path = os.environ['IFAO_READOUT_RECIPE'] or None
recipe = load_recipe(recipe_path)
if recipe.get('completion_marker'):
    assert Path(recipe['completion_marker']).read_text().strip() == '0'
ready = json.loads((out / 'readiness.json').read_text())
assert ready['status'] == 'ready_for_review_not_launched' and ready['cuts'] == recipe['expected_cuts']
assert ready['tasks'] == {'asr': recipe['expected_cuts']}
if recipe_path:
    assert ready['recipe'] == recipe and ready['recipe_sha256'] == sha(recipe_path)
    assert ready['updates'] == recipe['updates'] and ready['lr'] == recipe['lr']
    assert min(ready['finite_sampler']['batches_per_rank'].values()) >= recipe['updates']
assert ready['trainable_parameters'] == 388277760
assert ready['native_cpu_batch']['exact_final_target_and_EOS']
assert ready['model_dir'] == os.environ['IFAO_MODEL_PATH']
assert not (out/'hf').exists(), 'Fresh run only; do not reuse optimizer/checkpoints'
checks = {**ready['input_sha256'], **ready['model_file_sha256'], **ready['heldout_manifest_sha256'],
          **ready.get('code_sha256', {}),
          ready['manifest']: ready['manifest_sha256'],
          os.environ['IFAO_TRAIN_CONFIG']: ready['train_config_sha256']}
for path, expected in checks.items(): assert sha(path) == expected, path
print(f"Preflight passed: {ready['cuts']} ASR views; only asr_norm.weight/asr_head.weight expected trainable (388.277760M).", flush=True)
PY
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=1 trainer.start_batch=0 \
  trainer.num_epochs=1 "trainer.num_steps=$((updates - 1))" trainer.grad_accum_steps=1 \
  'trainer.frozen_modules=[audio_tower,language_model,projector]' \
  "trainer.optimizer.lr=$learning_rate" trainer.optimizer.weight_decay=0.0 "trainer.scheduler.eta_min=$learning_rate" \
  "trainer.valid_interval=$valid_interval" trainer.save_every_n=1 trainer.keep_last_k=2
