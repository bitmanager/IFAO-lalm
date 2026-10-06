#!/usr/bin/env bash
# Deployment recipe only: native IFAO resume, optimizer, scheduler and loss.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-phone2-context-stage5.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
root=/runs/dev-storage/ifao-data
previous="$root/runs/asr-short-context-v1"
export IFAO_RUN_DIR="$root/runs/asr-phone2-context-v1"
export IFAO_MODEL_PATH="$previous/hf"
export IFAO_TRAIN_CONFIG="$root/phone2-context-stage5/train.yaml"
export IFAO_VALID_CONFIG="$root/phone2-context-stage5/validation.yaml"
test -s "$root/phone2-context-stage5/summary.json"

while ! test -f /runs/ifao-short-context-stage4.exit; do
  sleep 15
done
test "$(cat /runs/ifao-short-context-stage4.exit)" = 0
test -s "$previous/epoch-4.pt"
# Validate the exact prepared data immediately before creating the new run.
python - <<'PY'
import hashlib
import json
from pathlib import Path
import yaml

root = Path('/runs/dev-storage/ifao-data/phone2-context-stage5')
summary = json.loads((root / 'summary.json').read_text())
assert summary['status'] == 'ready' and summary['cuts'] == 627389
assert summary['duplicate_cut_ids'] == 0
assert hashlib.sha256((root / 'train.jsonl.gz').read_bytes()).hexdigest() == summary['manifest_sha256']
config = yaml.safe_load((root / 'train.yaml').read_text())
assert len(config) == 1 and config[0]['manifest'] == str(root / 'train.jsonl.gz')
assert hashlib.sha256((root / 'validation.yaml').read_bytes()).hexdigest() == summary['validation_config_sha256']
for path, expected in summary['validation_manifest_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
PY
mkdir "$IFAO_RUN_DIR"
cp -a "$previous/hf" "$IFAO_RUN_DIR/hf"
ln -s "$previous/epoch-4.pt" "$IFAO_RUN_DIR/epoch-4.pt"
cd /ifao-context-review/lalm
# Native resume restores optimizer/scheduler (expected epoch5 LR 5e-5).
# Do not replace that state with a CLI learning-rate override.
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=5 trainer.num_epochs=5 \
  trainer.mixed_precision=bf16 data.sampler.max_tokens=2000 \
  trainer.save_every_n=1 trainer.valid_interval=1000 trainer.keep_last_k=2
