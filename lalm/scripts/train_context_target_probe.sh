#!/usr/bin/env bash
# Isolated 500-update diagnostic using the unchanged native trainer and data.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-context-target-probe.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
root=/runs/dev-storage/ifao-data
previous="$root/runs/asr-short-context-v1"
export IFAO_RUN_DIR="$root/runs/asr-context-target-probe29000"
export IFAO_MODEL_PATH="$previous/hf"
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$root/short-context-stage4/validation.yaml"
test -s "$previous/eval-checkpoints/checkpoint-29000.pt"
python - <<'PY'
import hashlib
from pathlib import Path

root = Path('/runs/dev-storage/ifao-data')
expected = {
    root / 'context-target-v1/train.jsonl.gz': 'aef8c9a5811033c9fbfd0ad70f24e1946a97558a22ebe0b661ec5fcb53c3a1de',
    root / 'short-context-stage4/validation.yaml': 'e7322428ce50ba4d6b76c50625b0cccd844ba3fcccd0e4caf3b816419bbbdafc',
}
for path, digest in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
PY
mkdir "$IFAO_RUN_DIR"
cp -a "$previous/hf" "$IFAO_RUN_DIR/hf"
ln -s "$previous/eval-checkpoints/checkpoint-29000.pt" "$IFAO_RUN_DIR/checkpoint-29000.pt"
# 83 training pairs, 7 conditions, two task views; repetition is not new audio.
cat > "$IFAO_TRAIN_CONFIG" <<EOF
- name: context-target-train-fit-probe
  manifest: $root/context-target-v1/train.jsonl.gz
  hours: 1.9729333333333334
  weights: 10
EOF
cd /ifao-context-review/lalm
# Native resume advances the epoch scheduler once: expected LR 5e-5.
# Native stop condition is global_step > num_steps, hence 29000 -> 29500.
# This diagnostic does NOT replace the continuing main run or the demo.
CUDA_VISIBLE_DEVICES=1 python -u train.py --config-name gigaam_context \
  trainer.start_batch=29000 trainer.start_epoch=5 trainer.num_epochs=5 \
  trainer.num_steps=29499 trainer.mixed_precision=bf16 \
  trainer.valid_interval=50000 trainer.save_every_n=1 trainer.keep_last_k=2
