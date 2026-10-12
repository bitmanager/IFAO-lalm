# Qwen transcription with causal self-history

## Semantic judge (validation only)

`benchmark_semantic_judge.py` submits the frozen per-recording metric CSV to
OpenRouter `openai/gpt-6-luna` (not 5.6). It uses Gemini as a user-selected
pseudo-reference, hides recognizer names, and deterministically shuffles A/B/C
per audio SHA. All three recognizers receive a rating in the same request.
The rubric distinguishes preserved meaning (0), minor changes (1), substantial
meaning changes (2), and potential action-changing/unusable transcription (3).
These are text-only risk judgments, not observed failures of an actual agent.

Outputs preserve quotes, uncertainty, raw requests/responses, mapping and billed
cost. Evidence quotes must occur in the input; whitespace differences alone are
allowed. Unsupported quotes fail validation rather than being counted. Restart
reuses matching valid results and can revalidate already captured responses;
`--retry-failed` explicitly resubmits remaining invalid responses, archiving them.
No alternate model, recognizer, judge vote or automatic pseudo-label selection is
used. A partial run exits nonzero, and rejected assessments are excluded from
aggregate rates with coverage reported. Reference doubts require listening.

## Runtime audit: remaining history contamination

`benchmark_qwen_runtime_audit.py` runs controlled native HF inference without
changing weights. It checks audio replacement leaves every non-audio embedding
unchanged, all embeddings are finite, and audio placeholders are replaced.
It compares history/no-history/silence and repeats the original generation after
other inputs. A separate uncached forward receives the same generated prefixes:
its next-token argmax is compared with cached generation at each position.
Setting `generate(use_cache=False)` alone is **not** this test: Transformers
forces caching for decoder-only `inputs_embeds` generation.

On four deliberately selected failing clips, all replacement checks passed,
repeated generations were identical, and uncached argmax matched all 75 generated
tokens. This bounds the diagnosis to those clips; it does not certify the full
runtime. With history the checkpoint still copies earlier conversation, including
on silence. Without history it recovers the address request and “алло”, largely
recovers the cryptocurrency payment sentence, but still misrecognizes another
short clip. Removing history is therefore not a complete ASR fix.
When independent transcript hypotheses replace the audio in the same prompt,
the native text path repeats all four hypotheses correctly with and without
history. This isolates a failure of audio-conditioned transcription under this
recipe, rather than a general inability to follow the text instruction.

Independent AGY Gemini 3.8 blind listening confirmed speech in all four clips.
One isolated-clip Gemini transcription disagreed with its full-call transcription
on the last word, so neither output is human gold. The checkpoint was optimized
for agent-response KL, not transcription. These results must not be interpreted
as an ASR accuracy gain or fixed by silently substituting another recognizer.
Runtime artifacts are stored outside git under
`qwen-kl1532-history-v3/runtime-audit-v3` and `agy38-blind-debug-v1` on drive2.

## v3: explicit transcription task and separated reference context

Full-call testing showed v2 could still replace the age statement with an older
transcript. On identical 181.0–187.4s audio, the no-history control returned
the age and question correctly. A stricter transcription instruction and an
explicit reference-context boundary before the audio also returned that content
correctly, with history retained. v3 uses this latter format uniformly, without
per-segment fallback, substituting another ASR, or deleting repeated text.

Six complete diagnostic calls were checked. The marketing call still has a
short mistaken repetition and lexical errors: this is not a claim that the
checkpoint now has perfect ASR accuracy. Long exact matches to history produce
a review warning, not an automatic rejection or a rewritten transcript.
`no_history` is an explicit diagnostic mode; it is not mixed into v3 results.

## v2: avoid replaying conversation as transcription

The first recipe reproduced a failure on the marketing call: a 53.1–55.8s
audio clip transcribed correctly alone, but with chat-role history copied
the previous user transcript. Long outputs also copied agent text verbatim.
This happened before Label Studio import.

The default recipe now merges adjacent speech spans across gaps of at most
2s, keeping a 30s maximum and never crossing a change of past agent context.
History is a reference block in the single system message; the current audio
is the only user message. Older history is still trimmed to the native 8192
token prompt budget. No model weights, decoder, or audio encoder change.
Result identity includes the new history mode and adapter hash. Explicit
Label Studio supersession archives old candidates and preserves human work.
`complete` only means generation completed, not that transcription is correct.

`prepare_qwen_history.py` is a format adapter for `benchmark_qwen_calls.py`.
It reads the exact Label Studio eligibility keys, cached native Silero spans,
source call/asset manifests and a checkpoint owner's existing probe specification.
It neither runs a model nor changes ASR, VAD, decoding or training.

```sh
python3 lalm/prepare_qwen_history.py \
  --plan /data/labelstudio/native-vad-gemini-ru-import-plan.full.json \
  --vad /data/labelstudio/silero-full-corpus-audit.json \
  --probe /data/special/qwen-kl1532-transcription-experiment.json \
  --source prod /data/prod-asr-benchmark-20260912-20261011-v1 /data/prod-asr-benchmark-20260912-20261011-v1/review-20261012T101026.csv \
  --source test /data/test-asr-benchmark-20260912-20261011-v1 /data/test-asr-benchmark-20260912-20261011-v1/review-20261012T101512.csv \
  --output /data/qwen-history/new-manifest.json
```

Run preparation on the host where the source `user_audio_path` values exist.
All input hashes are retained, including the native VAD configuration/version.
The adapter verifies each mono 16 kHz WAV's SHA and duration. Output must be a new
file: an already submitted inference manifest must not be rewritten.

Native VAD's rounded second boundaries are approximate speech boundaries, not
word alignment. The adapter adds 100 ms on each edge, clipped at the midpoint
between adjacent native spans and at file boundaries. Samples do not overlap.
For the current 640 eligible keys every resulting segment is shorter than 30 s.
An unexpected longer span fails explicitly and requires ready native segmentation
before preparation; it is never silently clipped or split from GigaAM text.

Original `client` text is excluded completely. Its recorded timestamp is not a
verified transcription-commit time and could reveal words in the current or
future audio. Segment histories contain only earlier original `agent` messages.
The existing inference runner adds its own previously completed segment
transcripts, available at the previous segment's end. It records the exact
history sent and any messages dropped for its context budget.

An ambiguous multi-call audio alias or invalid source clock holds **all original
agent messages**. These cases still use self-history and carry explicit flags.
The canonical identity is `(environment, media_id, user_sha256)`; a selected
`call_id` is only a provenance carrier for an ambiguous alias. Source audio/agent
clock and channel role remain separately marked as unverified where applicable.
An agent event timestamp is not independently verified as the completion of its
spoken message, so this benchmark does not establish strict causal gain from
agent history without further timing review.

The monthly set has 640 canonical keys but 629 unique audio hashes, including
11 prod/test duplicate pairs. The 630-key first run excludes 10 within-test
multi-call aliases; those 10 run with self-only history afterward. The full
`учусь` control (LS123119) is outside the monthly set and retains separate
accounting. The earlier LS123118 commit probe used a different history recipe.

These outputs are model candidates, never a human transcript gold. Self-history
can propagate transcription mistakes. Compare output quality through listening
and annotation; do not turn agreement between models into a WER claim.
