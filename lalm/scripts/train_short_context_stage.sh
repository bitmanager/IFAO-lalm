#!/usr/bin/env bash
# Deployment recipe: native IFAO resume, loader, optimizer, loss and checkpoints.
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-short-context-stage4.exit' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
root=/runs/dev-storage/ifao-data
previous="$root/runs/asr-phone-youtube-v1"
export IFAO_RUN_DIR="$root/runs/asr-short-context-v1"
export IFAO_MODEL_PATH="$previous/hf"
export IFAO_TRAIN_CONFIG="$root/short-context-stage4/train.yaml"
export IFAO_VALID_CONFIG="$root/short-context-stage4/validation.yaml"
test -s "$root/short-context-stage4/summary.json"

for attempt in $(seq 1 240); do
  if test -f /runs/ifao-asr-phone-youtube-train.exit; then break; fi
  sleep 15
done
test "$(cat /runs/ifao-asr-phone-youtube-train.exit)" = 0
test -s "$previous/epoch-3.pt"
mkdir "$IFAO_RUN_DIR"
cp -a "$previous/hf" "$IFAO_RUN_DIR/hf"
ln -s "$previous/epoch-3.pt" "$IFAO_RUN_DIR/epoch-3.pt"
cd /ifao-context-review/lalm
# Resume restores the optimizer/scheduler too; do not pretend a CLI LR changes it.
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=4 trainer.num_epochs=4 \
  trainer.save_every_n=1 trainer.valid_interval=1000 trainer.keep_last_k=2
