# Contextual overlap staging set

`lalm/prepare_context_overlap.py` adapts existing reviewed contextual cuts to
the PR2 `remix_foreground_pilot.py` interface. The existing mixer renders
`Cut.mix` / `MixedCut.load_audio`; no new DSP, TTS, teacher inference, trainer,
decoder or model is introduced. Every clean/mixed recording has two native
task views: foreground transcript (`asr`) and the original teacher response
(`answer`). Both preserve the foreground's system prompt and text history.

This is a **first-active-voice diagnostic**, not a solution to finding the
user's speaker from history. Mixtures identify the voice that enters first;
the background starts at least 0.26 seconds later, regardless of loudness.
Clean controls retain the ordinary task instruction. Mixed ASR views retain
the exact native transcription instruction and append the same first-voice
policy used by answer views. Background text is never inserted into either
conversation or target. The final ASR and answer labels stay different.

## Fixed source capacity and split

Sources, inside `nemo-asr-ru-20261005`:

* `/ifao-context-data/reviewed-v2-prepared-train.jsonl.gz`: 253 original cuts,
  with 159 eligible foreground turns containing history and teacher answers;
  these foreground turns total **951.68 seconds**.
* `/ifao-context-data/reviewed-v2-prepared-validation.jsonl.gz`: 18 original
  cuts, with 12 eligible foreground turns totaling **64.96 seconds**.

The 171 fixed pairs use all eligible foreground turns (1,016.64 seconds,
0.2824 hours of distinct foreground audio). They reuse existing synthesized
Russian audio and existing teacher labels. Reviewed source text does not
certify acoustic transcription accuracy or factual correctness of every
teacher answer. No new teacher answers were generated or corrected.

`assets/context_overlap_topics.json` records the manually reviewed broad
topics of the 68 source dialogues. `assets/context_overlap_pairs.jsonl`
selects different broad topics, source groups and voice IDs for each pair.
Each role keeps its original split. Both roles are checked for cross-split
source groups, recording IDs, exact case-folded transcripts and decoded-PCM
hashes. WAV container hashes separately track artifact integrity. Renderer
metadata hashes and participant/channel voice mappings
verify declared voice IDs. This guards this source set, not every external
corpus. Voice IDs **do overlap across splits**; this is not speaker-disjoint.
One short validation foreground requires a longer background tail; its fixed
recipe records that choice. No truncation removes the retained background.

All three SNR levels (+3, 0, -3 dB target/interference) use the same pair and
onset. At least 50% of target active frames must overlap background activity.
Activity uses the existing 20 ms relative-RMS proxy, not annotated speech
boundaries. Lhotse SNR is a whole-track energy ratio; the mixer also records
the measured ratio over simultaneously active frames. Isolated input audio,
target/interference stems, native Lhotse mix JSON, gains, timings, WAV hashes,
transcripts, full original cuts and source-manifest hashes are retained.

## CPU-only reproduction

In the existing exp container, use `CUDA_VISIBLE_DEVICES=''`,
`PYTHONPATH=/deps:/review/vendor:/olddeps`, `OMP_NUM_THREADS=2` and
`OPENBLAS_NUM_THREADS=2`. The tokenizer is loaded from local files only.
Output directories must not exist.

```sh
python lalm/prepare_context_overlap.py \
  --source-manifests /ifao-context-data/reviewed-v2-prepared-train.jsonl.gz \
    /ifao-context-data/reviewed-v2-prepared-validation.jsonl.gz \
  --recipe assets/context_overlap_pairs.jsonl \
  --output /runs/dev-storage/ifao-data/context-overlap-v1 \
  --tokenizer /runs/ifao-balalaika-export/epoch-2
```

For the 12-pair smoke use `assets/context_overlap_smoke_pairs.jsonl` and a
different output path. The ready full root maps to dev-1
`/mnt/local/drive1/ifao-data/context-overlap-v1`. Native absolute paths in
these manifests refer to the exp container bind.

Completed export (CPU only):

| Split | Foreground pairs | Audio variants | Audio hours | Native task cuts |
| --- | ---: | ---: | ---: | ---: |
| Train | 159 | 636 | 1.09830556 | 1,272 |
| Fixed validation | 12 | 48 | 0.07884444 | 96 |
| Total | 171 | 684 | 1.17715000 | 1,368 |

The two task views reuse each WAV, yielding 2.3543 task-exposure hours; they
do not double the unique audio. There are 513 mixtures and 171 clean controls.
The selected roles cover 58 train and 7 validation source groups. All 684
audio variants were reread and decoded, all 1,368 labels/history/system values
matched their foreground sources, and all durations passed the existing
0.5–30 second training filter. Minimum target active-frame overlap is
0.56779661; maximum stem-additivity error is 0; maximum mixture peak is
0.90000010. These are structural/numerical checks, not model-quality scores.

`summary.json`, `readiness.json` and `mixed/audio-qc.json` retain the results
and manifest/WAV hashes. Source and stem WAVs plus native mix JSON live under
`source/` and `mixed/`. The execution log is at sibling
`context-overlap-adapter-code/full.log`. The smoke set lives at
`context-overlap-v1-smoke`, including `native-cpu-batch-qc.json`. Two initial
smoke attempts are retained separately; they exposed and preceded the fixed
custom-field/method-name collision (`cut.custom["split"]`, not `cut.split`).

`train.jsonl.gz` and `validation.jsonl.gz` contain both tasks; the eight
`context-overlap-{clean,snr+3,snr+0,snr-3}-{asr,answer}.jsonl.gz` files contain
only the fixed validation set. `eval-asr.yaml` and `eval-answer.yaml` group
conditions for native evaluation. Run the former with `+asr=true`; run the
latter in ordinary answer mode. No training configuration references these
manifests. **Every task cut remains `training_eligible=false`.**

CPU tests cover task/history preservation, source-group/PCM leakage rejection,
and actual native mixer/export execution. A real eight-cut smoke batch also
passed the unchanged `LALMDataset` and `LALMProcessor`: four ASR flags, exact
last-assistant labels, and history masked from labels. No model weights or
GPU inference were loaded for these structural checks. Subsequent independent
source ASR and native epoch-2 diagnostics are documented in
[CONTEXT_OVERLAP_QC.md](CONTEXT_OVERLAP_QC.md), including quarantine and an
independent clean-source eligibility mask. Human listening and a complete
answer-fact rubric remain pending. This small set
is an addressed supplement to real phone/SOVA data, not a large standalone
training epoch.

## Next recipe: target selected from context

The existing first-voice mixer intentionally fixes one ordering. A subsequent
format adapter can express two orders using native Lhotse alone: delay the
background with `foreground.mix(background, offset_other_by=d, snr=s)`, or
delay the foreground with `background.mix(foreground, offset_other_by=d,
snr=-s)`. In the latter form swap the exported track-to-role mapping; Lhotse
defines SNR relative to its first track, so the sign must reverse to preserve
target/interference SNR. Native `pad`/`truncate` can bound a shared timeline
if needed. Save exact track roles, offsets and both stems in either order.

Retain the foreground's history/teacher answer and unrelated background,
keep source-group holdout and identical pairs across onset/SNR conditions,
and replace the first-voice qualification with a consistent instruction to
continue the interlocutor from the history. Clean controls retain their
ordinary instruction. This requires a small recipe/format extension, not
new DSP or an architecture change. It is now implemented as a separate
optional export described in [CONTEXT_TARGET.md](CONTEXT_TARGET.md); the
first-voice artifacts and results above remain unchanged.
Text history identifies topic/context but supplies no acoustic enrollment:
generic utterances, same-topic intrusions, and two equally plausible voices
remain ambiguous. Both-order evaluation should report those limits and
background-word intrusions separately from ASR WER and answer-fact scores.
