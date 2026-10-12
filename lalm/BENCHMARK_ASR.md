# Russian production/test ASR benchmark

`benchmark_langfuse_asr.py` is an orchestration/format adapter. It uses the native
Langfuse CLI for API requests, ready FFmpeg for channel extraction, our existing
SIP preparation runner for native GigaAM RNNT decoding, and native AGY `view_file`
for Gemini. It does not implement ASR, model inference, forced alignment, or API
pagination. It does not change training or deploy models.

The reproducible default cohort is 12 September–11 October 2026, Europe/Moscow:
`2026-09-11T21:00:00Z <= call_start < 2026-10-11T21:00:00Z`.
Production and test remain separate. Russian is selected by explicit language
metadata or the trace's `ru` tag. All raw exports remain available for audit.
`prepare` also accepts `--start` and `--end` for a later rolling cohort; these
are inclusive/exclusive ISO-8601 call-start bounds. This only re-filters source
exports already fetched for that interval. It does not schedule or fetch a new
window automatically, so each refresh must retain its raw export and date range.

Audio, model outputs and review files belong on dev/exp, outside this repository.
The current data roots on dev are:

```
/mnt/local/drive2/prod-asr-benchmark-20260912-20261011-v1
/mnt/local/drive2/test-asr-benchmark-20260912-20261011-v1
```

## Source exports

Use official `@langfuse/cli@1.2.4 --api-version auto api observations list --all --json`.
The explicit `auto` is required by the self-hosted Langfuse v3 instance; without
it, requests hit the CLI's v4 default route. The tested v3 endpoint accepts time
bounds, but does not accept the v4-only `--fields` and `--is-root-observation`
options. Root/tag filtering must therefore happen with a supported native v3
export or after fetching observations; do not assume a command copied from v4
will work. It returns
an envelope with `body.data`; observation IO/metadata can be JSON strings.
The adapter rejects truncated exports and duplicate observation IDs.

Recording spans start at upload, not call start. Export recordings with adjacent
date padding, then join root observations by `traceId`. For roots, combine native
filter conditions: `tags any of [call]`, `isRootObservation = true`, and both
explicit date bounds. Native `--filter` takes precedence over basic query fields.
Use `core,basic,time,metadata,trace_context` for roots: broad HTTP/LLM IO is large
and unnecessary. Recording export needs IO to retain media and online turns.
Export `speech.asr` separately with model/provider/phase/status/timing metadata.

Credentials are supplied to the native CLI container from the existing service
environment in memory. Never put keys or signed media URLs in the repository,
review CSV, terminal output, or prompts.

```
python lalm/benchmark_langfuse_asr.py prepare \
  --root "$DATA_ROOT" --environment prod \
  --recordings "$DATA_ROOT/raw/recording-observations-boundary.json" \
  --roots "$DATA_ROOT/raw/call-root-observations.json" \
  --start 2026-09-11T21:00:00Z --end 2026-10-11T21:00:00Z
```

Prepare retains missing recordings and empty online results in `calls.jsonl` and
`cohort.json`. Audio assets are deduplicated; distinct calls are not. Playground,
campaign, copied and simulated provenance remains visible. The initial synthetic
transport trigger `Начни разговор` is not transcribed user speech.
Report `recording_source_status` distinguishes an absent recording span from an
existing span with no media link. Neither is a failed download. A nonempty media
marker in an unsupported format fails preparation loudly instead of silently
discarding that audio.

## Ready ASR runners

```
python lalm/benchmark_langfuse_asr.py download --root "$DATA_ROOT" \
  --api-container prod-asr-benchmark-api-20261012
```

For the current self-hosted MinIO URL, explicitly supply `--media-proxy` pointing
to an SSH loopback forward. The original signed Host header is preserved. This
does not change the Langfuse/MinIO services. `--ffmpeg` can specify the ready
FFmpeg in our existing container. Existing originals and derived audio must all
match their stored hashes before a download is reused.

The adapter emits `$DATA_ROOT/giga/data/calls.raw.jsonl` and an empty
`control_truth.jsonl` for the unchanged existing `prepare_sip.py` runner:

```
python /path/to/existing/prepare_sip.py --root "$DATA_ROOT/giga" \
  --model /path/to/gigaam-sip-readable.ckpt --device cpu --threads 8 --batch-size 8
```

Pinned SIP weights SHA256:
`607e71d4d9fa87fadbdd779357fc3d1d064e1c273e7080cafd325ce474ba591e`.
Pinned existing runner SHA256:
`5a1d2023f130278f4c1b8205409ba7a47df94bf4011b5f0cf43fcffafb7828f8`.
The runner calls native GigaAM forward/RNNT decoding. Its native word emissions
are approximate boundaries, not original online commits. Empty ASR failures
remain in the coverage ledger. Stereo CH0 is the recorder-declared user channel;
geometry alone does not verify that role. Mono fallback remains explicitly
unverified. Native model/script/audio cache provenance must be checked on resume.

```
python lalm/benchmark_langfuse_asr.py gemini --root "$DATA_ROOT" \
  --workspace /path/to/empty/agy-workspace --variant blind
python lalm/benchmark_langfuse_asr.py gemini --root "$DATA_ROOT" \
  --workspace /path/to/empty/agy-workspace --variant trace_assisted
python lalm/benchmark_langfuse_asr.py report --root "$DATA_ROOT"
```

AGY uses a fresh process per recording and pins `gemini-3.8-flash-low`. Never use
`--continue` or share a conversation across calls. Blind transcription sees only
user-channel WAV. Trace-assisted transcription separately sees online context,
which may contain errors or leak answers. These are two information conditions
of the same model, not independent recognizers. Blind jobs reuse identical audio;
context jobs are per call. Both compare offline full-call information, not a fair
equal-context online commit test.

The current trace-assisted payload is exactly `origin` plus the entire recorded
`online_turns` history (roles, text and timestamps). It includes later user turns
and agent answers; it is full-trace-assisted reconstruction, not past-history-only
recognition. It does not include the agent's actual system prompt or task config.
For this cohort, 347 prod and 638 test call jobs have processable audio; 22 prod
and 44 test jobs have empty recorded histories. Keep those empty-history jobs
separate when comparing the effect of context. Do not interpret an improvement
with full trace as a causal improvement available to the live recognizer.

Accept an AGY result only with native success, one turn, a matching model, a
successful `view_file` of the exact allowed audio and validated structured output.
External ASR/commands/extra file reads invalidate the result. Preserve failed
attempts and stderr; bounded retry is only for explicit transient errors.

### Direct Gemini audio API (access blocked in this environment)

`benchmark_gemini_api.py` is a separate transport adapter to the native
OpenRouter audio API, pinned to `google/gemini-3.8-flash`. It sends the verified
user-channel WAV as `input_audio`, requests native JSON Schema, records original
responses/billed usage, and disables provider fallbacks. Results go into
`gemini-openrouter/<variant>/`, separate from AGY. They are not merged into AGY
report columns or represented as the same inference condition.

The existing evaluation key was tested in memory. The native smoke request was
denied with HTTP 403, `Access denied by security policy`; no bulk API job ran.
The adapter stops on authorization/billing denial. Do not try alternate keys or
network routes to bypass that policy. AGY access was separately restored after
the user's subscription upgrade and a successful native audio smoke test.
That does not restore the denied direct API route.

Once access is restored, test one asset before launching a corpus:

```
python lalm/benchmark_gemini_api.py --root "$DATA_ROOT" \
  --media-id "$MEDIA_ID" --max-cost 0.10
```

Credentials come from `EVAL_OPENROUTER_API_KEY` in the process environment, never
from repository files. `--max-cost` is a soft stopping threshold per root/variant,
including previous native responses, not a hard provider spending limit. A single
request can exceed the remaining threshold, and already in-flight requests can
finish afterward. Missing/nonfinite/negative billed usage stops new requests and
restarts until billing is reconciled. Recorded access denial blocks reruns unless
`--retry-restored-access` is explicitly supplied after access has been restored.
This API path has contract
tests, but audio inference has not passed an authorized native smoke test yet.

Primary protocol documentation:
[audio inputs](https://openrouter.ai/docs/guides/overview/multimodal/audio),
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

## Review and metrics

Reports are timestamped snapshots so manual edits in prior files are preserved.
Both Gemini variants expose completion status. Captured provider filter refusals,
failed attempts, attempts awaiting validation, unavailable audio and uncompleted
jobs remain distinct. Quota-blocked attempts retain a separate status; statuses
describe the latest captured attempt, which can predate an in-flight retry.
A native envelope saying `SUCCESS` does not override a
filter refusal or make an invalid multi-turn result acceptable. Status reporting
does not retry rejected requests or accept their partial transcripts.
Keep reviewed annotations keyed by `call_id`; do not treat a new snapshot as
carrying earlier manual decisions. Report similarity is a sequence-match diagnostic,
not WER. Include missing/failed jobs in coverage, not just completed transcripts.
Reports include both online-vs-GigaAM and GigaAM-vs-Gemini similarity, original
online model labels when trace telemetry exists, and listening priority. A native
`Empty ASR` is distinct from an unstarted job; listen before classifying it as a
recognition error. Model labels such as `v3_rnnt` do not establish an identical
checkpoint hash to the pinned offline SIP weights. Missing telemetry remains
visible. Language/script differences can distort text similarity even for the
same pronunciation. Priority is only triage, never an error or winner label.
`online_text` comes from the recorded client history. `online_text_source`
distinguishes histories with matching ASR observations from those whose ASR
origin is unverified. Absence of telemetry does not prove typed input; presence
does not prove a one-to-one match between every history turn and an ASR commit.
Do not attribute all history/audio mismatches to recognition: input mode, capture
and commit/drop policy also require checking.

Listen to disagreements and a representative sample of agreements. Review names,
numbers, negations, hallucinated/omitted words, repetitions and speaker mixups.
Logical plausibility with the history is useful for triage but is not proof of
what was spoken. Neither Gemini nor offline GigaAM is human ground truth. WER/CER
requires independently checked reference transcripts. Original recording clocks,
onset timestamps, interrupted agent text and channel normalization can differ
from the original online audio; retain that limitation in comparisons.

Contract checks:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p test_benchmark_langfuse_asr.py -v
```
