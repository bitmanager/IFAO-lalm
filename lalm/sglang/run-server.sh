#!/usr/bin/env bash
set -euo pipefail
# Run on exp x4-02 only. Native checkpoint and all cache remain on that host.
bench=/mnt/local/drive2/checkpoint-response-bench-kliter2-epoch8-v1/sglang
runtime=/mnt/local/drive2/ifao-qwen38-sr-20261010
gpu=GPU-e501f17d-17ef-5e19-53b2-ec8c42258758
model="$runtime/cache/huggingface/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
test "$(hostname)" = exp-srv-rtx6000x4-02
# Fail rather than start alongside another large model on our allocated card.
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
test "$free" -gt 85000
mkdir -p "$bench/cache"
docker run -d --name asr-bench-sglang-epoch8 --gpus "device=$gpu" \
  --user "$(id -u):$(id -g)" --group-add 26403 --ipc=host \
  -p 127.0.0.1:18139:18139 \
  -e HOME=/cache -e HF_HOME=/cache/huggingface \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/bench-no-mps \
  -v /mnt/local/drive2:/mnt/local/drive2:ro \
  -v "$bench/cache:/cache" \
  -v "$bench/qwen3_vl.py:/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py:ro" \
  --entrypoint python lmsysorg/sglang:v0.5.21 \
  -m sglang.launch_server --model-path "$model" \
  --tokenizer-path "$runtime/data/tokenizer" \
  --json-model-override-args '{"language_model_only":true}' \
  --dtype bfloat16 --disable-radix-cache \
  --mem-fraction-static 0.80 --context-length 12288 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --host 0.0.0.0 --port 18139
