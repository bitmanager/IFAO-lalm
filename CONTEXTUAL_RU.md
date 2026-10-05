# Contextual Russian audio-prefix data

Based on upstream `rorizzz/IFAO-lalm` at
`3a8d0d7f12f9a5cbd4a91f3d3fd9269eacebb331`.

For each example, the teacher and student receive the same system prompt and
preceding user/assistant messages. The teacher receives the current transcript;
the student receives only the corresponding audio cut. The teacher's response
is the sole supervised assistant turn. Earlier assistant messages and padding
are masked. This uses the upstream causal language-model loss, not TARS/RL or an
auxiliary ASR head. The training loop is unchanged.

## Data contract

A Lhotse `MonoCut` selects the current user channel, start and duration from a
long recording. Its single supervision contains the current transcript. Custom
fields are:

- `history`: preceding text messages, with `role` and `content`;
- `system`: the exact system prompt used by teacher and student;
- `source_group_id`: original conversation identity, shared across its cuts;
- `split`: original recording-level train/validation/test assignment.

Use accepted user-turn boundaries from the application when available. A turn
can contain several VAD segments; do not treat every silence as a new request.
Keep internal pauses. Do not replay the live wall-clock VAD faster than realtime
to generate offline boundaries. No state or KV cache is carried between training
examples: the relevant history is explicitly included in every example.
The teacher currently has an 8192-token context budget. This patch does not yet
implement a rolling history window for hour-long conversations; exceeding that
budget fails explicitly. Long-recording provenance and short audio cuts do not
by themselves establish support for a two-hour conversation context.

The included MultiTalk adapter reuses existing, locally rendered synthetic
dialogues as an integration fixture. It preserves channels, timestamps, split,
and prior turns; consecutive user fragments are one audio span. These are
synthetic turn bounds, **not production commits**. Word alignment is unnecessary
for this prefix objective. The adapter rejects unsupported/malformed examples;
it does not invent history for unrelated ASR clips.

## Prepare and generate targets

Run on the server holding the recordings. No per-cut WAV copies are produced.
Choose the real deployment system prompt; `lalm/configs/system_ru.txt` is the
neutral Russian experiment prompt, not a booking-agent persona.

```bash
python lalm/prepare_multitalk_context.py \
  --metadata-root /data/multitalk/pilot-v1 \
  --system-file lalm/configs/system_ru.txt --output-dir /data/context-cuts

python lalm/prepare_conversation.py \
  --input_manifest /data/context-cuts/train.jsonl.gz \
  --output_manifest /data/context-train.jsonl.gz --tokenizer /models/qwen \
  --teacher-input /data/teacher-train.jsonl

CUDA_VISIBLE_DEVICES=3 python utils/batch_inference_vllm.py \
  --model_path /models/qwen --scp_file /data/teacher-train.jsonl \
  --dataset_size 128 --out_dir /data/teacher-responses \
  --num_workers 2 --queue_max 64 --max_seqs 8 --tensor_parallel_size 1

python lalm/prepare_conversation.py \
  --input_manifest /data/context-cuts/train.jsonl.gz \
  --output_manifest /data/context-train.jsonl.gz --tokenizer /models/qwen \
  --responses /data/teacher-responses
```

Repeat separately for validation/test without moving conversations across
splits. Teacher response files retain the exact input messages; import checks
them against the current manifest. Empty/truncated answers and mismatched
prompts fail. History that exceeds the teacher context budget fails instead of
silently truncating the latest request. Russian punctuation, case and numbers
are preserved. Rendering uses the selected tokenizer's chat template.

The source generator now accepts `messages` in addition to its original caption
format. Generation stays with upstream vLLM; tensor parallelism can use one GPU
for Qwen3-4B-Instruct-2507 without moving other workloads.

## Verification and remaining integration

```bash
QWEN_MODEL_PATH=/models/qwen python -m pytest tests/test_contextual_data.py -q
```

Tests use the real Qwen tokenizer and native Lhotse cuts. They check audio
channel/offset selection, merging user fragments, excluding future turns,
teacher/student history equality, current-answer-only loss and prompt mismatch
rejection.

On 2026-10-05, the four adapter tests passed with the real
Qwen3-4B-Instruct-2507 tokenizer. The existing 16-dialogue fixture exported 36
train, 5 validation and 6 test cuts. Native vLLM generated all 36 train answers
in BF16 on one RTX PRO 6000 Blackwell GPU, and all were imported with matching
teacher messages. This is a small integration check, not a training run.

This change prepares **contextual data**, not a complete GigaAM deployment.
Upstream IFAO still needs its GigaAM audio-tower/input adapter before these cuts
can train our frozen GigaAM + Qwen stack. The existing NeMo ASR run is separate
and does not train this objective. The small synthetic fixture validates the
pipeline; it is not a corpus-size or model-quality claim. The prepared 105-hour
Balalaika ASR corpus must not acquire fabricated conversational history.
