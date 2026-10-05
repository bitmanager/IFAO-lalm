# Russian foreground speech pilot

This isolated pilot asks for the designated main utterance while another
voice speaks, including equally loud and louder interference. It is a small synthetic curriculum, not a benchmark for
identifying an arbitrary desired speaker. There is no enrollment input and no
architecture or main-training change.

`assets/foreground_pilot_v1.jsonl` contains 16 explicit phrase pairs: 12 train
and 4 validation. Each produces a clean example, an overlapping-speech example,
and a background-only example with an empty target and silent target waveform.
Phrase strings and voice IDs do not cross splits; all variants of a pair stay
together. Two voices alternate foreground/background roles in each split.
Validation has different voice IDs, but voice-bank labels alone are not an
independent speaker-identity audit. Numbers are written as spoken words.

The adapter calls the **existing** `Teachers.synthesize` client from the user's
Lychee pipeline. It uses only already published voice IDs from the running
`bitmanagerai/Qwen3TTS-v3.7-mix50` API, saves the model and voice-bank revisions,
and does not launch or change serving models. Mixing, offsets, resampling and
relative SNR are provided by **Lhotse `Cut.mix`**. No custom waveform mixer or
speech generator is introduced.

The existing `conversation_router.mixtures` CLI was also inspected. Its fixed
LibriSpeech dev/test scenario design includes louder backgrounds and arbitrary
hidden targets, so Lhotse is the applicable upstream builder for this pilot.

Example using the existing CPU environment on dev-1:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/local/drive2/conversation-router-poc-ib-20260911/envs/poc/bin/python \
  lalm/prepare_foreground_pilot.py \
  --recipe assets/foreground_pilot_v1.jsonl \
  --teacher-client /home/se@bitmanager.ai/lychee-ru-pipeline/adapters.py \
  --tts-url http://10.20.0.11:19099 \
  --output /mnt/local/drive1/ifao-data/foreground-pilot-v1
```

The output directory must not exist. Partial output is retained on failure.
The adapter imports only the existing client's module, never the student codec.
Its upstream Python dependencies must already exist; it installs nothing.

Outputs:

- `raw-source/`: original 24 kHz TTS WAVs and client metadata.
- `audio/<pair>/`: aligned 16 kHz FLOAT WAV mixture, target, interference and
  silence; the exact Lhotse mix description; plain transcript files.
- `pilot.jsonl`, `train.jsonl`, `validation.jsonl`: audio paths, target text,
  speaker reference IDs, timing, requested SNR, split and condition.
- `sources.jsonl`: input scripts, source audio and hashes.
- `provenance.json`: model/voice revisions, upstream client and adapter hashes.
- `audio-qc.json`, `summary.json`: additivity, amplitude, durations and counts.

Requested background SNRs are 9/12/15 dB; start offsets are 0/0.25/0.65/1.2
seconds. Lhotse's SNR uses source energy, so the report also gives the measured
aligned-stem SNR. Both sources share any final peak-headroom adjustment. FLOAT
WAV retains additive stems without independent PCM16 quantization.

The scripts are intended transcripts, not independently verified acoustic gold.
ASR and human listening status remain explicit. The output marks
`training_eligible=false` until reviewed. Background-only examples deliberately
contain speech despite their empty targets; this narrow task follows the
foreground definition and should not be interpreted as generic VAD. No codec,
room, real-call noise, or independent speaker enrollment is simulated. Empty
targets must not be passed to the current `asr_cut` adapter, which rejects them.

## Verified dev-1 artifact

`/mnt/local/drive1/ifao-data/foreground-pilot-v1` was generated using TTS model
revision `384ffff264f0407498f3ca7138871b9cf03f69f6` and voice-bank revision
`860451b106bc799d5235f392a9380b509c7c2d31`, with Lhotse 1.33.0.

There are 48 examples (36 train, 12 validation), 16 source pairs, 32 raw TTS
sources and 16 empty targets. Each complete example lasts 2.64–4.72 seconds.
The 16 distinct mixtures total 56.72 seconds; the manifest with paired controls
totals 170.16 seconds (0.047267 hours). Raw source speech totals 92.96 seconds.
These durations describe different representations and must not be summed as
independent material.

Independent verification reread all 64 final WAVs, checked 32 original source
hashes and all plain transcripts, and confirmed silent targets, disjoint phrase
and voice-ID splits, and an unchanged serving-model revision. Maximum saved
WAV additivity error was `1.49e-8`, maximum peak `0.4571`, and measured aligned
stem SNR ranged from 7.79 to 16.33 dB. Evidence is in
`saved-audio-verification.json` and `audio-qc.json` beside the manifests.
`component-gains.json` also records the gains measured against each resampled
original source, including the background start offset.

The configured legacy teacher ASR endpoint at `127.0.0.1:19081` was unavailable.
The later ASR check used the existing GigaAM upstream batch evaluator on exp-1
GPU3 with our `/model.ckpt` instead; no requests went to unrelated ASR canaries.
Human listening remains pending and the pilot is not approved for training.
An initial two-source attempt exposed Lhotse 1.33's list-shaped unmixed output;
it remains separately preserved at `foreground-pilot-v1-attempt1` and is not
part of the final manifests.

## GigaAM check of v1

Evidence is in `foreground-pilot-v1/asr-qc-gigaam-v1/metrics.json` and
`cases.jsonl`, alongside the exact input TSVs, predictions and logs. The
checkpoint SHA-256 is
`607e71d4d9fa87fadbdd779357fc3d1d064e1c273e7080cafd325ce474ba591e`.
Inference used unchanged upstream `train_utils/eval.py`, batch size 16, and
the container's existing PyTorch 2.11 environment. The process excluded the
incompatible `/gpu` PyTorch 2.9 override without modifying either environment.
Text comparison uses GigaAM's `normalize_raw_text`, with no digit verbalization.

| Input | Records | Normalized result |
| --- | ---: | --- |
| Original clean sources | 32 | 0/209 word errors |
| Saved 16 kHz clean targets | 16 | 0/101 word errors |
| Mixtures | 16 | 28/101 word errors (27.72%); 8 exact |
| Background only, empty target | 16 | 16 nonempty outputs, 108 inserted words |
| Exact digital silence | 16 | 16 empty outputs, zero inserted words |

Every mixture hypothesis preserved the complete target as an exact normalized
prefix. The errors were appended competing-speech tails. The background-only
result concerns the task's target-selection rule: the recognizer transcribed
the competing speech, rather than hallucinating words on silence. WER is left
null for all-empty references, instead of presenting upstream's zero-denominator
percentage as a meaningful metric. These are GigaAM baseline results, not the
IFAO auxiliary head. Original generation manifests retain their initial
`asr_qc=pending` snapshot; the dated QC sidecar supplies the completed check.

## Overlapping speech at three levels

`lalm/remix_foreground_pilot.py` reuses the same 32 source WAVs and upstream
Lhotse mixing to compare each of the 16 unrelated phrase pairs at target to
background SNRs of +3, 0 and -3 dB. There are 48 mixtures and 16 clean controls.
The 12 train pairs and 4 validation pairs keep their original disjoint phrase
and voice-ID splits. Sources are checksum-verified and referenced in v1; no
new TTS requests are made.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/local/drive2/conversation-router-poc-ib-20260911/envs/poc/bin/python \
  lalm/remix_foreground_pilot.py \
  --source-pilot /mnt/local/drive1/ifao-data/foreground-pilot-v1 \
  --output /mnt/local/drive1/ifao-data/foreground-pilot-v2-overlap \
  --snr-db 3 0 -3
```

The target is the **first active voice**, regardless of loudness. The competing
voice begins at least 260 ms after target activity starts. Each mixture must
have simultaneous activity during at least half of the target's active frames.
Activity is an explicit QC energy proxy: 20 ms RMS frames exceeding -30 dB of
each source's peak frame RMS, not a phonetic speech annotation. The export
records offsets, observed activity onsets, overlap duration/fraction, requested
SNR, measured SNR over simultaneous activity, waveform gains and checksums.
Isolated target/interference and the exact Lhotse description accompany every
mixture. Source-energy SNR and SNR during overlap can differ.

First-voice selection requires a consistent task instruction or conversation
context. Equal or louder competing voices cannot in general identify an
arbitrary desired speaker without such information; this pilot adds no
enrollment or model architecture. Background-only negatives remain in v1:
first-entrant identity alone cannot label an isolated competing voice as
non-target. Neither version is connected to main training.

The generated v2 directory contains 64 examples (48 train / 16 validation),
160 saved WAVs and 32 referenced original sources. Mixture variants total
160.14 seconds, with 45.60 seconds of clean controls; 205.74 seconds in the
manifest is **0.05715 hours**, not new independent speech. Independent reread
verified every saved WAV, source hash, transcript and split. Saved additivity
residual was at most `5.96e-8`; maximum peak was `0.8318`. Simultaneous activity
lasted 1.04–1.94 seconds, covering 61.7–87.4% of target activity.

The same unchanged GigaAM evaluator ran all 48 mixtures on exp-1 GPU3:

| Requested target/background SNR | Normalized WER | Exact target transcripts |
| --- | ---: | ---: |
| +3 dB | 45/101 = 44.55% | 4/16 |
| 0 dB | 74/101 = 73.27% | 2/16 |
| -3 dB | 104/101 = 102.97% | 0/16 |

These harder mixtures cause target substitutions/omissions and competing-word
intrusions, beyond v1's appended tails. For example, at 0 dB the target
`нет я не подтверждаю этот заказ` becomes
`нет я не подтверждаю закрой окно на кухне`. Competing speech is
`поставь чайник и закрой окно на кухне`. WER may exceed 100% when insertions add
to substitutions and deletions. This is a baseline recognizer without a
first-voice instruction, not a measured improvement from training.
The v2 `asr-qc-gigaam-v1/{metrics.json,cases.jsonl}` sidecars contain results,
per-level metrics, exact predictions and immutable evaluator/checkpoint hashes.
All examples remain ineligible for main training; listening review is pending.

## Additional source shard 01

The next staging shard uses the same `Teachers.synthesize` and Lhotse adapters.
Its 64 scripts are verbatim first user turns from existing DataDesigner
dialogues accepted by the existing DuplexOmni quality filter. Acceptance here
is automatic (`accepted=true`, `failed=false`, matching `hallucination=false`),
not human correction or acoustic gold. Source files, hashes, dialogue IDs,
turn indices, original context and voice IDs are recorded in
`assets/foreground_scale_shard01.jsonl`.

The two inspected pools provide 54 + 118 original-train first turns satisfying
the conservative standalone filter (35–130 characters, 7–18 words, no Latin
letters or digits). Seven additional qualifying turns are original holdout
and excluded. These 172 candidates span 54 + 65 source groups; repeated
scenario seeds must remain together in any larger export. This is available
text capacity, not promised unique speaker-hours. Counts and source hashes are
in `assets/foreground_scale_text_sources.json`.

All 64 selected source scenarios were originally train. The later seed-group
audit found three scenario groups shared by our proposed train and validation.
The overlap export therefore excludes **train** pairs `fgs01-008`, `fgs01-011`
and `fgs01-015`, preserving the validation set. The complete audit is in
`assets/foreground_scale_shard01_split_audit.json` and beside both manifests.
Retained texts, dialogue IDs, source seed groups and voice IDs do not cross
splits. Retained source groups also have no intersection with original
holdout groups, and no new exact phrase repeats the first pilot.
The eight voice IDs are six for train (`anastasia-1`, `valera-1`,
`male-ru-14`, `male-ru-17`, `male-ru-21`, `male-ru-25`) and two reserved for
validation (`ekaterina-1`, `male-ru-37`). Voice IDs are not a biometric audit.

Artifacts on dev-1:

- `/mnt/local/drive1/ifao-data/foreground-scale-shard01-source`: all 64 original
  24 kHz TTS sources, 362 seconds (0.100556 hours), recipe and provenance.
  The existing source adapter also emits intermediate +9 dB examples and
  empty-target controls; these are staging and are not the final overlap set.
- `/mnt/local/drive1/ifao-data/foreground-scale-shard01-overlap`: 29 retained
  pairs, 58 referenced sources (326.56 seconds), 87 mixtures and 29 clean
  controls, 84 train / 32 validation records. Mixture variants total 543.84
  seconds, clean controls 165.84 seconds, or 0.197133 manifest hours. Variants
  are correlated and must not be counted as independent new speech.

The final export was produced with:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /mnt/local/drive2/conversation-router-poc-ib-20260911/envs/poc/bin/python \
  lalm/remix_foreground_pilot.py \
  --source-pilot /mnt/local/drive1/ifao-data/foreground-scale-shard01-source \
  --output /mnt/local/drive1/ifao-data/foreground-scale-shard01-overlap \
  --snr-db 3 0 -3 \
  --exclude-pair fgs01-008 fgs01-011 fgs01-015
```

`--exclude-pair` is an explicit manifest filter. It rejects unknown IDs and
an empty retained set and records exclusions in provenance. It adds no new
synthesis, mixing or inference mechanism. The fixed recipe preserves the
exact synthesized inputs; exclusion happens only during overlap export.

Independent reread checked all 290 saved WAVs, transcripts and 58 source hashes.
Maximum peak is 0.77145; maximum saved additivity residual is `2.98e-8`.
Simultaneous activity lasts 2.20–5.08 seconds and covers at least 57.46% of
target activity. Target identity remains first active voice, never loudness.
The competing source comes from a separate topic/context; this is unrelated
synthetic speech, not natural dialogue or a verified real-room overlap corpus.

The same unchanged GigaAM evaluator checked all 64 new raw sources, then the
87 final mixtures on exp-1 GPU3. Evidence is under the overlap directory's
`asr-qc-gigaam-v1/`, including per-example cases and exact predictions.

| Input | Normalized WER | Exact transcripts |
| --- | ---: | ---: |
| All 64 original sources | 9/844 = 1.07% | 57/64 |
| 58 retained sources | 8/766 = 1.04% | 52/58 |
| +3 dB mixtures | 129/381 = 33.86% | 6/29 |
| 0 dB mixtures | 265/381 = 69.55% | 2/29 |
| -3 dB mixtures | 379/381 = 99.48% | 0/29 |

Clean disagreements include inflections and rare-word spelling, requiring
listening to distinguish TTS errors from ASR errors. The mixture result is an
unconditioned ASR baseline, not target-selection training progress. All
records remain `training_eligible=false`; human listening is pending. There
is no added telephone codec, echo, room noise, enrollment or main-training
connection.

Voice `male-ru-06` returned `empty_upstream_audio` twice and was replaced by
the existing `male-ru-25` ID before the final generation. The incomplete first
attempt and five successful voice probes remain separately in
`foreground-scale-shard01-source-attempt1` and `foreground-scale-voice-probe`;
they are excluded from final manifests. Serving models were not changed.
