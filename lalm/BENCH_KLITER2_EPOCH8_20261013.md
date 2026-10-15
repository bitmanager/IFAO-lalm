# kl_iter2 epoch-8: full benchmark execution

Status: launched, results pending. Do not interpret this document as a completed evaluation.

Checkpoint: `/mnt/local/drive2/ifao-qwen38-sr-20261010/runs/response-kl-large-fresh-8ep-v1/epoch-8.pt`.
SHA-256: `9b7bd2a3d5137a0796e7e5792cd055caf8224b413d636e8951942459b6ff6ef9`.
Epoch 8, step 15990; use `projector`, not `projector_ema`. No ASR head is in this checkpoint.
The projector has 40,905,728 parameters. Qwen/GigaAM and native inference are unchanged.

User allocated exp-srv-rtx6000x4-02 GPU1:
`GPU-e501f17d-17ef-5e19-53b2-ec8c42258758`.
All inputs, outputs, logs, model cache and evaluation state remain on that host.

Run directory: `/mnt/local/drive2/checkpoint-response-bench-kliter2-epoch8-v1`.
The pending manifests remain as pre-allocation provenance; execution uses `*.gpu1.json`.

## Frozen evaluation

- Same 204 full-WAV response cases as KL1532, four branches, no history.
- Same 629 unique monthly recordings, 3698 planned segments, history format v3.
- Same greedy native generation, BF16 backbone, prompts and pseudo-references.
- ASR uses the native Qwen language head with a transcription prompt, not a separately trained ASR head.
- GPT-6 Luna through stock promptfoo, same two rubrics for each track.
- Stock jiwer 4.0.0 corpus micro WER/CER, raw and normalized. Normalization remains lowercase, ё→е, punctuation/underscore to spaces, whitespace collapse; numbers are unchanged.
- Empty/truncated candidates stay in primary metrics. The complete-only subset is secondary.
- Gemini references remain pseudo-labels, not human gold.

`export_checkpoint_asr.py` only joins native results to the frozen CSV and invokes jiwer.
Its raw/normalized scores and all counts reproduced the previous 629-call report exactly.

## Durable jobs

Records/logs/exit codes are in `/mnt/local/drive2/ifao-qwen38-sr-20261010/jobs`:

1. `bench-kliter2-epoch8-answers204-gpu1-v1`
2. `bench-kliter2-epoch8-asr629-gpu1-v1` — starts only after successful response inference.
3. `bench-kliter2-epoch8-response-judge-v1` — grades completed responses concurrently with ASR inference.
4. `bench-kliter2-epoch8-asr-metrics-judge-v1` — joins and scores completed ASR, then runs semantic grading.

Each judge job imports a separate saved report into the existing promptfoo UI only after zero technical errors and complete coverage. Previous reports are preserved.

UI: http://exp-srv-rtx6000x4-02.tail8f0274.ts.net:15500/

An inference failure stops dependent jobs rather than producing an apparently complete report.
ASR resumes per call/segment; the 204-case response adapter does not resume and must not be restarted casually.
