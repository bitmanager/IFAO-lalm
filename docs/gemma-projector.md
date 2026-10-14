# Frozen Gemma audio encoder pilot

Reuse `train_projector.py`, manifests, Qwen tokenizer, projector, CE and DDP.
Only the audio tower changes. No QUAN weights, reader, CTC or layer mixing.
The tower returns the last Conformer sequence (1024 channels), before its
1536-channel output projection and before pooling. Both backbones stay frozen.

Requires Transformers exposing `embedding_gemma2` and `gemma4`. Test in an
isolated environment before deploying to the existing GigaAM runtime.
The official feature extractor runs on CPU with truncation disabled; this
avoids duplicating Mel code, but throughput must be measured later.

Example for a future separate run (not launched by this PR):

```sh
export QWEN_MODEL=/path/to/qwen
export PROJECTOR_RUN=/path/to/new-gemma-run
export TRAIN_DATA_CONFIG=/path/to/train.yaml
export VALID_DATA_CONFIG=/path/to/valid.yaml
python lalm/train_projector.py --config-name train_projector_gemma
```

Use prepared ASR conversations with transcription targets. The Gemma config
inherits existing batch/duration/epoch settings: override them for a small pilot.
Response KL is disabled in both trainer and data; the objective is transcription
CE. GigaAM projector initialization is rejected. No new training loop.

CPU validation: `pytest tests/test_gemma_adapter.py`. Tiny random official
Gemma model, no downloaded weights or GPU. Checks length boundaries, masks,
batch independence, placeholder lengths, config reload and projector gradients.
Full Qwen/Gemma weights, Russian quality and GPU throughput remain untested.

Validation in this change: Python syntax and git diff checks passed; length
arithmetic checked across 63,680 waveform sizes against the official frame-end
and stride-mask rules. CPU model tests could not run because dependency
downloads stalled in the isolated local environment. No runtime success claimed.
