# History-selected foreground, both onset orders

`prepare_context_overlap.py --context-target --clean-qc-mask ...` is a small
extension of the existing adapter and native Lhotse mixer. It creates a
separate `context-target-v1` export. The first-voice dataset, frozen validation
and completed epoch-2 results at `context-overlap-v1` are not overwritten.

The frozen independent source-QC rule selects **83 train pairs**: both
isolated roles must exactly match the existing GigaAM-normalized transcript.
The full **12 validation pairs** are preserved, along with a separate
**five-pair** mask. Neither new inference nor native epoch-2 predictions enter
this selection. Mask entries now also bind source cut IDs and decoded PCM
hashes, so altered source audio cannot silently inherit an admission decision.
The mask's pair decisions are unchanged from the preceding source-QC audit.

The foreground retains its original system, history, transcript and teacher
answer. Mixed task views use the common qualification:

> Целевой собеседник — пользователь из истории диалога. Учитывай только его
> реплику, продолжающую эту историю; игнорируй нерелевантную фоновую речь.
> Не выбирай собеседника по громкости или по тому, кто заговорил первым.

ASR retains its exact native transcription instruction before this
qualification. Answer mode uses its ordinary task plus this qualification;
the shared policy does not ask the ASR view to generate a conversational
answer. Clean controls retain neutral original task instructions. The
background's transcript is never injected into model input or target.

Each pair has one clean control and six mixtures: target-first and
background-first at +3/0/−3 dB target/interference SNR. The first active source
precedes the other by at least 0.26 seconds in each direction. At least half
the foreground's active frames must overlap the unrelated background in
both directions. Preflight passed all 95 pairs; no pair substitution or
relaxed overlap threshold was needed.

Rendering uses only existing native operations. Target-first calls
`target.mix(background, offset_other_by=d, snr=s)`. Background-first calls
`background.mix(target, offset_other_by=d, snr=-s)`, then maps the native
second track back to the target stem. The sign reversal preserves the
declared target/interference ratio. Native `perturb_volume` supplies the
existing common clipping protection; no new DSP is introduced. Metadata
retains native track roles, actual gains, onset offsets, active overlap,
requested whole-source SNR and measured overlap-only SNR. Initial loudness
can differ by onset order because the native first track is the energy
reference; target identity does not depend on loudness. The normalized
source control is shared across the two directions.

## Reproduction and outputs

Run inside the existing exp container with CPU-only
`CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=/deps:/review/vendor:/olddeps`,
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`:

```sh
python lalm/prepare_context_overlap.py \
  --source-manifests /ifao-context-data/reviewed-v2-prepared-train.jsonl.gz \
    /ifao-context-data/reviewed-v2-prepared-validation.jsonl.gz \
  --recipe assets/context_overlap_pairs.jsonl \
  --clean-qc-mask assets/context_overlap_clean_qc_mask.jsonl \
  --context-target \
  --output /runs/dev-storage/ifao-data/context-target-v1 \
  --tokenizer /runs/ifao-balalaika-export/epoch-2
```

The root maps to dev-1 `/mnt/local/drive1/ifao-data/context-target-v1`.

Completed CPU export:

| Split | Source pairs | Audio variants | Audio hours | Native task cuts |
| --- | ---: | ---: | ---: | ---: |
| Train selection | 83 | 581 | 0.98646667 | 1,162 |
| Full fixed validation | 12 | 84 | 0.14002778 | 168 |
| Total | 95 | 665 | 1.12649444 | 1,330 |

There are 95 clean controls and 285 mixtures per onset direction. Distinct
foreground duration is 550.72 seconds; two task views reuse audio, producing
2.25298889 task-exposure hours. Do not count the masked validation view as
additional data.

All 665 rendered WAVs were decoded. Both isolated stems were checked against
their exact source samples at the recorded offsets. Maximum target/interference
SNR error is 0.00000169 dB; maximum reread stem-additivity error is 0.000000060.
Minimum foreground active overlap is 62.78% for target-first and 60.10% for
background-first. Minimum onset gap is 0.26 seconds, maximum mixture peak
0.90000015, and PCM/source-group split overlap is zero. Original first-voice
manifest hashes still match their prior readiness record.

All 1,330 final labels, systems and histories were checked against originals.
Ten CPU tests pass. A real native `LALMDataset`/`LALMProcessor` batch contains
all seven conditions and both tasks (14 examples), with seven ASR flags.
Decoded supervised tokens equal each final assistant target, history is
masked, and every target contains EOS token 151645. The processor ran on CPU
without model weights. Details: `readiness.json`, `native-cpu-batch-qc.json`
and `mixed/audio-qc.json`. New context-target model inference has not run.

Load **only this training manifest** for the next train composition:

```python
from lhotse import CutSet

train = CutSet.from_file("/runs/dev-storage/ifao-data/context-target-v1/train.jsonl.gz")
assert len(train) == 1162
assert all(c.custom["split"] == "train" and c.clean_source_qc["clean_qc_candidate"] for c in train)
assert {c.task for c in train} == {"asr", "answer"}
```

Useful cut metadata: `task`, `condition`, `onset_order`, `snr_db`,
`target_policy`, `mixture_pair_id`, `foreground_source_cut_id`,
`source_group_id`, `history`, `system`, `rendered_conversation` and
`clean_source_qc`. `onset_order` is null for the shared clean control.

`train.jsonl.gz` / `validation.jsonl.gz` contain both tasks. Each of
`eval-asr.yaml` and `eval-answer.yaml` contains seven conditions with all
12 fixed foregrounds. `eval-asr-cleanqc.yaml` and `eval-answer-cleanqc.yaml`
contain the corresponding independent five-pair diagnostics. Full validation
must be reported first. Conditions are clean and
`{foreground-first,background-first}-snr{+3,+0,-3}`.

Original selected cuts, isolated sources and provenance remain under
`source/`; rendered mixtures, both stems, clean audio, native Lhotse mix
JSON and audio QC are under `mixed/`. The execution log is the sibling
`context-target-adapter-code/export.log`. All outputs are separate staging
artifacts; `training_eligible=false` and `main_training_connected=false`.
The admitted **source-pair** selection does not itself connect or change
training, certify teacher-answer correctness, or assert model performance.

This condition uses text history and topic relevance, not acoustic speaker
enrollment. Generic responses or two contextually plausible voices remain
ambiguous. Different topics and source groups are paired, while speaker IDs
can cross train/validation. No TTS, teacher/model inference, decoder or
trainer was added or run for this export.
