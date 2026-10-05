# Russian foreground speech pilot

This isolated pilot asks for the designated main utterance while another,
quieter voice speaks. It is a small synthetic curriculum, not a benchmark for
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
ASR transcript checks and human listening therefore remain pending; the pilot
is not approved for training. No requests were sent to unrelated ASR canaries.
An initial two-source attempt exposed Lhotse 1.33's list-shaped unmixed output;
it remains separately preserved at `foreground-pilot-v1-attempt1` and is not
part of the final manifests.
