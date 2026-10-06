# Contextual Russian audio-prefix data

Based on upstream `rorizzz/IFAO-lalm` at
`3a8d0d7f12f9a5cbd4a91f3d3fd9269eacebb331`.

For each example, the teacher and student receive the same system prompt and
preceding user/assistant messages. The teacher receives the current transcript;
the student receives only the corresponding audio cut. The teacher's response
is the sole supervised assistant turn. Earlier assistant messages and padding
are masked. The original projector-only run uses the upstream causal
language-model loss. An optional intermediate ASR readout is described below;
neither path implements TARS/RL or hidden-state/KL alignment. The training loop
is unchanged.

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

## GigaAM integration

Install the upstream dependencies pinned in `requirements-gigaam.txt` into the
GPU environment. Auden's `lalm_hf` branch is required: its default branch has
incompatible precision, batch and scheduler-checkpoint interfaces. Auden and
the training loop are used without local patches.

`gigaam_adapter.py` calls the official GigaAM batched forward. It only adapts
packed waveforms, output shape/lengths and checkpoint configuration. The RNNT
head is discarded. Raw audio and spectrogram preprocessing stay FP32; the
official encoder uses FP16 autocast and FlashAttention, while Qwen uses BF16.
The Hugging Face waveform processor supplies padding, not a Wav2Vec2 model.
The supported checkpoint has uncentered 320-sample windows, hop 160 and
subsampling 4; other timing configurations fail instead of guessing lengths.

```bash
cd lalm
CUDA_VISIBLE_DEVICES=0 python build_model.py \
  --llm /models/qwen --encoder /models/gigaam-sip.ckpt \
  --output_dir /runs/ifao-gigaam-base --dtype bfloat16 \
  --projector_downsample_rate 2

CUDA_VISIBLE_DEVICES=0 \
IFAO_MODEL_PATH=/runs/ifao-gigaam-base \
IFAO_RUN_DIR=/runs/ifao-context-v1 \
IFAO_TRAIN_CONFIG=/data/context-v1-train.yaml \
IFAO_VALID_CONFIG=/data/context-v1-validation.yaml \
python train.py --config-name gigaam_context
```

Each data YAML is the upstream list of `{name, manifest, hours, weights}`.
The configuration selects three finite epochs, BF16, raw waveform input and
2000 tokens per batch. Encoder/LLM freezing, the two-layer projector, causal CE,
optimizer and training loop remain upstream. Only raw-input dtype handling in
the trainer is adapted; Qwen's packed attention path uses native SDPA. Validation
honors the configured bucket count, including small held-out datasets.

## Verification and limitations

```bash
QWEN_MODEL_PATH=/models/qwen python -m pytest tests/test_contextual_data.py -q
CUDA_VISIBLE_DEVICES=0 IFAO_MODEL_PATH=/runs/ifao-gigaam-base \
IFAO_MANIFEST_PATH=/data/context-train.jsonl.gz \
python -m pytest tests/test_gigaam_integration.py -q -s
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

The GPU test loads the actual saved model and real audio, verifies finite,
nonzero gradients on all four projector parameters and no gradients elsewhere,
and exercises raw-audio generation without the trainer's autocast context.
All five tests passed. The projector has 10,490,880 trainable parameters. A full
upstream trainer smoke run completed an optimizer step, validation and checkpoint
save. These are integration checks, not evidence of useful speech understanding.

The first larger experiment uses 688 available user turns from 84 rendered
dialogues. Eleven truncated teacher responses were explicitly quarantined.
Scikit-learn `GroupShuffleSplit(test_size=0.1, random_state=114514)` separated the
remaining examples by `source_group_id`: 599 training cuts (75 conversations,
0.716 audio hours) and 78 validation cuts (9 conversations, 0.105 hours).
Those two manifests subdivide the source's training split; the original pilot
validation/test remain separate. This is a small experiment, not a large corpus.

Audit correction: this pilot initially missed the source's existing quarantine
file. Two dialogues (`5bc8c6e653792899b03f`, `820fd23046e3d7a4934e`) contain
assistant facts not grounded in preceding user speech. They must be excluded
from subsequent training/validation manifests. The pilot's loss is only an
integration measurement, not a clean quality baseline.

For additional reviewed text, `lalm/prepare_reviewed_scripts.py` converts the
existing approval manifest and byte-identical scenario files into MultiTalk's
TTS input format. It verifies source hashes, source-group splits and alternating
user/assistant turns. Only spoken history is converted; hidden scenario text is
not inserted into the conversation. The existing MultiTalk worker/mixer and our
existing TTS API adapter render the audio; this file does not synthesize speech.

```bash
python lalm/prepare_reviewed_scripts.py --manifest /data/accepted.jsonl \
  --scenario-root /data/scenarios --output /data/scripts.jsonl
```

The first export passed on all 130 approved dialogues (68 source groups).
Do not merge them with earlier data without checking source-group overlap.

Native two-GPU launch uses `CUDA_VISIBLE_DEVICES=0,2 python -m
torch.distributed.run --standalone --nproc_per_node=2 train.py` with the same
configuration. Nsight Systems traces of short runs measured 0.363 s/step on one
GPU and 0.402 s/step on two at 2000 tokens per rank. Estimated aggregate token
throughput increased 1.87x (rank-0 throughput multiplied by world size; not exact
cross-rank accounting). A 4000-token budget was slower per token. These short
measurements exclude validation/checkpoint IO and include profiling overhead.

The upstream packed path constructs a dense 4D block-diagonal causal mask.
BF16 memory-efficient SDPA kernels were observed; this is not sparse attention
or cross-example KV-cache reuse. Teacher forcing predicts answer positions in
parallel. Removed the pilot's save-every-50-step overrides: full 24-GiB
checkpoints took about 10 seconds each. The config now inherits upstream
validation/save intervals; epoch checkpoints remain enabled.

Contextual training and the shared Russian system prompt are our adaptation,
not the exact instruction-free single-turn template from the IFAO paper.
Quality must be checked against correct transcript input, silence and mismatched
audio under identical history. The 105-hour Balalaika ASR corpus is separate and
must not acquire fabricated conversational history. The prior NeMo ASR run has
been stopped; its final step-3000 checkpoint is retained.

## Optional Qwen ASR readout

The readout is our integration, not a published IFAO/NVIDIA ASR branch. It uses
the existing Qwen transformer, tokenizer, Hugging Face causal CE and generation.
`add_asr_readout.py` strictly loads a native trainer checkpoint and copies Qwen's
final RMSNorm and vocabulary projection into independent trainable modules.
For Qwen3-4B this adds 388,277,760 parameters; it is not a tiny classifier.
The readout consumes hidden states after layer 35 of 36. Qwen, its native
answer head and GigaAM remain frozen; projector, ASR norm and ASR head train.

```bash
cd lalm
python add_asr_readout.py --checkpoint /runs/ifao-context-v2-ddp/epoch-3.pt \
  --output-dir /runs/ifao-context-asr-initial
```

This preserves trained model weights and starts a new optimization stage.
Optimizer/scheduler states are not carried across the changed parameter set.
The first stage uses native `gigaam_context` training with one epoch and
`trainer.optimizer.lr=0.0001`, with the new HF directory as `IFAO_MODEL_PATH`.
No full-backbone training or separate Transformer decoder is introduced.

`prepare_conversation.py --with-asr` adds a separate transcript example for
each answer example. Both retain preceding history and the source split. The
ASR example adds an explicit transcription instruction and uses the current
transcript as its target. The answer example never receives that transcript.
Independent packed causal segments prevent cross-task/example target leakage.
The loss is token-averaged CE over the two tasks, with a 1:1 example mix; no KL
term or additional loss weighting is currently used.

`task=asr` routes its packed sequence to the intermediate readout. At evaluation,
`generate(asr=True)` constructs a shallow native Qwen view sharing the first 35
layers and using the independent norm/head. It does not mutate the agent model
or implement a custom decoding loop. Native `evaluate_qa.py +asr=true` evaluates
transcript-task manifests using greedy generation and jiwer WER/CER. It reports
raw WER and lowercased/punctuation-stripped WER/CER separately; numbers are not
expanded and ё is not replaced. The head remains optional for agent inference.

The real-call adapter `prepare_calls_context.py` reads the existing stereo IVR
manifest. It checks channel permutations, hashes, splits and turn bounds, and
includes only complete preceding turns. These are approximate RNNT turn bounds,
not production VAD commits. Source train/dev remain train/validation; test and
control recordings are excluded. Roles are relative to the selected speaker,
not inferred operator/customer labels.

The first joint stage contains 2,555 unique training cuts (3.128 hours), comprising
846 earlier synthetic cuts and 1,709 real telephone cuts, doubled into 5,110
task examples. Validation has 688 unique cuts (0.849 hours) from 40 groups,
disjoint from 210 training groups. One invalid teacher response was quarantined.
Counting both tasks gives 6.255 training hours of exposure, not unique audio.
Telephone transcripts are GigaAM pseudo-labels: their WER measures agreement
with those labels and cannot establish superiority over GigaAM.

Validation: 27 tests passed, including a real GPU backward/generation test,
native checkpoint reload, packed-versus-native readout equality, empty-ASR
batch gradients for DDP, target isolation and unchanged agent modules.
The real test found gradients only on the projector and ASR norm/head:
398,768,640 trainable parameters. This verifies integration, not ASR quality.

The first one-epoch run completed 800 optimizer steps and saved
`/runs/ifao-context-asr-v1/epoch-1.pt`. On a fixed 31-cut panel (15 synthetic,
16 telephone, one cut per source group), normalized free-decoding WER changed
from 208.31% before training to 142.49% after training. WER above 100% comes
from insertions, including repetitive/hallucinated text. This is still poor ASR,
not evidence of usable transcription or superiority over the source recognizer.
The two prior agent examples still lose current-utterance facts; text-input
answers remain unchanged. Silence/wrong-audio controls are retained alongside
correct-audio outputs. No further epoch was launched automatically.

## Balalaika ASR stage

After explicit approval of the next stage, `prepare_asr_manifest.py` adapts the
existing Lhotse ASR corpus without synthesizing/copying audio or generating
teacher answers. It unwraps a single MonoCut from padding-only MixedCuts, checks
audio paths and transcript bounds, preserves the transcript and uses only the
ASR readout task. It never invents dialogue history. Three tests cover waveform
preservation, isolated transcript targets, mixed-speaker rejection and bounds.

```bash
python lalm/prepare_asr_manifest.py --input-manifest /data/prepared/train.jsonl.gz \
  --output-manifest /data/balalaika-train.jsonl.gz \
  --tokenizer /runs/ifao-context-asr-v1/hf \
  --system-file lalm/configs/system_ru.txt
```

The export contains 64,619 training cuts / 105.542 hours and 662 validation
cuts / 1.084 hours. The existing validation split hashes transcripts; source
recording/speaker separation is not established. It is an additional diagnostic,
not a replacement for source-disjoint contextual validation.

The next finite epoch consumes all Balalaika training cuts and three passes of
the 5,110 contextual task examples, using native `CutSet.mux(stop_early=False)`
to write a single shuffled-mixture manifest. This avoids upstream's multi-source
`stop_early=True` ending an epoch when the smaller source is exhausted. There
are 79,949 examples: 72,284 ASR and 7,665 answer tasks. Unique audio is about
108.67 hours; total task exposure with replay is 124.31 hours.

Native resume uses `trainer.start_epoch=2 trainer.num_epochs=2`, restoring both
model and optimizer from `/runs/ifao-context-asr-v1/epoch-1.pt` at step 800.
The new run directory is `/data/ifao-runs/asr-balalaika-v1` on the separate
data volume. Its `epoch-1.pt` is a symlink to the retained source checkpoint.
GPU assignment remains 0,2, BF16 and 2000 tokens/rank. Intermediate full checkpoint
saves are disabled with `trainer.save_every_n=1000000`; native epoch saving and
periodic validation remain enabled, avoiding multiple 30-GiB snapshots filling
the volume. This is one approved larger epoch, not an unattended infinite run.

That epoch completed at global step 6578 and saved `epoch-2.pt`. Greedy
normalized WER on the fixed contextual 31-cut panel improved from 142.49% to
12.47%; on the 64-cut Balalaika diagnostic panel it improved from 217.27% to
15.29%. These small panels retain the pseudo-label and split limitations above.
They do not demonstrate superiority over the original GigaAM recognizer.
Two agent-response examples retain more current-utterance facts, but still
invent details. The ASR readout also hallucinates on silence; this remains an
unresolved validation failure, even though ordinary speech improved.

## Phone and YouTube continuation

Two format adapters reuse `prepare_asr_manifest.asr_cut`, native Lhotse and
the existing ASR task. Neither changes the model, decoder or training loop:

- `prepare_openstt.py` reads the official OpenSTT phone training CSV, audio and
  transcripts. It counts exact duplicate rows once, rejects conflicting rows
  and validation subsets, and supports explicit held-out audio exclusions.
- `prepare_youtube_balalaika.py` reads the existing FLAC/JSON tar shards and
  punctuated ASR transcripts. Whole podcast sources in the supplied held-out
  voice list are excluded. It preserves each clip without invented history.

Five adapter tests passed, covering target isolation, waveform preservation,
official validation rejection, duplicate handling and whole-source exclusion.

The continuation contains 206,016 OpenSTT phone cuts (186.250 hours, after
removing 27,852 duplicate CSV rows), 32,739 local YouTube cuts (63.435 hours),
the previous 105.542-hour Balalaika set and six passes through the contextual
task examples. The YouTube export excluded 1,768 clips from 12 held-out sources
and 185 invalid-text clips. Both new sources have automatic transcripts, not
human gold labels. Audio-content deduplication across different corpora has not
been established; 358.355 hours is the sum before task replay, not a claim of
globally unique recordings. With task replay, exposure is 392.759 hours across
318,704 ASR and 15,330 answer examples.

Native finite `CutSet.mux(stop_early=False)` consumes the complete manifest.
Training resumes the previous model and optimizer at step 6578 with
`trainer.start_epoch=3 trainer.num_epochs=3`, BF16 and GPUs 0,2. Existing
contextual and Balalaika validation panels remain unchanged. The run and its
new checkpoints are on dev storage, exposed inside the exp container as
`/runs/dev-storage/ifao-data/runs/asr-phone-youtube-v1`. Data are likewise read
over NFS; container processes need the authorized developers group (26403).
This live-container NFS mount must be restored after container replacement.

Older projector-only `context-v2` epoch-1/2 checkpoints were copied to dev's
`/mnt/local/drive1/ifao-checkpoint-archive/context-v2`, SHA256-verified, and only
then removed from exp's root volume. The current continuation checkpoint was
retained. Background-speaker mixtures are a separate pilot in PR #2 and have
not been added to this training manifest.

For subsequent launches, the contextual config now sets `save_every_n: 1`:
native Auden saves after every validation (the inherited interval is 1000
global steps), retaining two step checkpoints. Do not pass the earlier
`trainer.save_every_n=1000000` override on future launches. The already running
epoch uses its in-memory configuration and is unaffected. Auden has no native
on-demand save signal or live config reload; killing/restarting this run would
discard progress since epoch 2. A fresh step checkpoint can be evaluated using
the existing evaluator on a separate GPU without changing the training loop.

## Staged SOVA RuDevices format conversion

`lalm/prepare_sova_parquet.py` reads already downloaded train parquet shards
with `audio.bytes` and `transcription`, exports the original WAV bytes in
64-row batches, and creates `Recording`/`SupervisionSegment` objects passed
to the existing `prepare_asr_manifest.asr_cut`. It requires `pyarrow` in the
preparation environment; it does not download data or load an ASR model.
The staging manifest must contain verified checksums, file sizes, row counts
and durations. The adapter checks sizes/totals and retains staging checksums
in provenance; it does not repeat the full parquet checksum scan.

```bash
CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false \
python lalm/prepare_sova_parquet.py \
  --source-root /runs/dev-storage/ifao-data/sova-rudevices \
  --output-dir /runs/dev-storage/ifao-data/sova-asr-v1/prepared \
  --exclusions-json /runs/dev-storage/ifao-data/sova-asr-v1/audit/exclusions.json \
  --tokenizer /runs/ifao-context-asr-initial \
  --system-file /runs/dev-storage/ib-offline/slam-asr-data/langfuse-pstn-v1/prepared-v1/system.txt
```

Use a fresh output directory. The completed manifest is named `train.jsonl.gz`;
an interrupted run leaves `train.incomplete.jsonl.gz`, which must not be used
for training. `train.yaml` is a standalone source entry, not an edit to the
active training configuration. Export paths must be visible under the same
absolute path to the future trainer.

The exclusion JSON has `texts`, `ids`, and `sha256` string lists and an optional
`sources` audit list. Text comparison ignores case, punctuation, whitespace,
and the distinction between е/ё. This deliberately also excludes coincident
short phrases; it does not establish shared recording identity. IDs and
existing audio hashes are checked separately. Word alignment tokens should
not be supplied as whole-utterance references. Exact duplicate audio bytes
inside train are exported once. Exclusions and their source row coordinates
are retained in `excluded.jsonl`.

Byte-identical audio with conflicting normalized transcripts is quarantined
in every variant, including an earlier copy already written to the temporary
manifest. A held-out transcript also excludes later/earlier copies with the
same audio hash. The final pass uses native Lhotse manifest filtering and
does not re-export or segment audio.

SOVA's dataset card describes manually annotated Russian speech and CC-BY-4.0
licensing. This conversion retains those annotations without independent
re-annotation. No history, target-speaker label, background-speech label, or
segmentation is invented. The staged parquet has no usable utterance path or
speaker ID, so IDs are stable shard/row coordinates with audio SHA256. Only
train is staged; upstream validation/test and speaker-disjointness cannot be
verified from these local files.
