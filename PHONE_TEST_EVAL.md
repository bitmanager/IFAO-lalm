# Frozen phone ASR evaluation

`lalm/prepare_phone_eval.py` adapts the existing phone-test TSV to native Lhotse
cuts through the unchanged `prepare_asr_manifest.asr_cut` helper. It preserves
all 1,943 examples, their original reference strings and waveforms, including
29 examples shorter than the training CLI's 0.5-second lower bound. No train
manifest, model, decoder or inference implementation is changed.

The source is `Malecc/asr_calls_2_val`, revision
`36f9769d1548bfa9282b615737e5c29e29149198`, split `test`. The adapter verifies
the frozen TSV SHA-256 before export. Cut IDs use `phone-test-000000-asr` and
carry the original row, relative path, waveform file hash, source revision,
`evaluation_only=true` and `training_eligible=false`. The native evaluation
configuration uses result name `phone1943-canonic`.

The original exp-1 data are at
`/mnt/local/drive1/ib-offline/slam-asr-data/phone-eval`.
The copy accessible to IFAO is at dev-1
`/mnt/local/drive1/ifao-data/phone-test-eval-v1`, mapped inside the existing exp
container to `/runs/dev-storage/ifao-data/phone-test-eval-v1`.
Existing absolute `/data` bindings point elsewhere, so this copy avoids
changing container mounts. Audio, TSV, source Parquet and the original GigaAM
predictions/metrics were copied; original files remain unchanged.

All 1,943 Parquet `text` and `transcript` values equal the TSV references.
All decoded WAV samples equal the corresponding Parquet audio, at 16 kHz.
There are 1,943 distinct waveform file hashes. Actual sample-count duration is
1.14606375 hours; the TSV's millisecond-rounded durations total 1.14606389 hours.
Audio durations span 0.36–18.03 seconds. Dataset reference labels are preserved;
independent documentation of human correction was not located, so this audit
does not certify human gold or speaker/source independence across corpora.

## Reproduce the CPU format export

In `nemo-asr-ru-20261005`, with `CUDA_VISIBLE_DEVICES=''` and the existing
`PYTHONPATH=/deps:/review/vendor:/olddeps`:

```sh
python lalm/prepare_phone_eval.py \
  --tsv /runs/dev-storage/ifao-data/phone-test-eval-v1/phone-test.tsv \
  --audio-root /runs/dev-storage/ifao-data/phone-test-eval-v1 \
  --output-dir /runs/dev-storage/ifao-data/phone-test-eval-v1/native \
  --tokenizer /runs/ifao-balalaika-export/epoch-2 \
  --system-file lalm/configs/system_ru.txt
```

The output directory must not exist. The export contains
`phone-test.jsonl.gz`, `eval.yaml` and `summary.json`. The dataset root also
contains `reference-audit.json`, `source-parquet-index.jsonl`, and the unchanged
baseline under `predictions/phone-test/models/gigaam-sip/preds.jsonl`.
An initial export snapshot remains separate in `native-initial-snapshot`.

## Compare the same normalization

The **same frozen GigaAM hypotheses** score differently under the existing
normalizers. Neither reference text nor native IFAO scoring has been modified.

| Normalizer applied to both references and predictions | GigaAM WER | GigaAM CER |
| --- | ---: | ---: |
| Existing GigaAM `normalize_raw_text` | 17.8653% (2,109 / 11,805 words) | 7.50439% |
| Existing IFAO `jiwer` composition | 20.3219% | 8.02161% |

GigaAM folds `ё→е`, splits Unicode dashes, retains word-internal apostrophes,
and keeps alphanumerics/whitespace. IFAO lowercases, removes punctuation,
collapses whitespace and strips. On this set, 319 normalized reference rows
differ; normalized hypotheses do not. Therefore compare native IFAO output
with **20.3219%**, or rescore both systems with GigaAM normalization before
comparing against **17.8653%**. Do not compare the two normalizer outputs as
model improvements. No digit verbalization is performed.

Baseline checkpoint SHA-256:
`607e71d4d9fa87fadbdd779357fc3d1d064e1c273e7080cafd325ce474ba591e`.
The original evaluator was GigaAM commit
`7447938d791c4f3e643386ee22c33777004293a5`, batch 64, FP16 encoder, flash disabled.

## Native IFAO evaluation

After GPU scheduling, reuse the existing exported weights without re-export:
The native LALM runtime uses
`PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash`
(existing PyTorch 2.9.0+cu130 and its matching FlashAttention). This differs
from the GigaAM batch-QC runtime. CPU import and Lhotse WAV loading were checked
before launch; dependencies were not installed or changed.

```sh
python /ifao-context-review/lalm/evaluate_qa.py \
  exp_dir=/data/ifao-runs/asr-balalaika-v1 \
  checkpoint.model_dir=/runs/ifao-balalaika-export/epoch-2 \
  checkpoint.epoch=2 \
  data.test_data_config=/runs/dev-storage/ifao-data/phone-test-eval-v1/native/eval.yaml \
  data.max_duration=40 data.num_workers=2 \
  +asr=true dtype=bf16 max_new_tokens=256
```

Use only the assigned physical GPU. The maximum duration is the native batch
duration budget, not a new item filter. The existing evaluator emits
`qa_results-phone1943-canonic-epoch-2.txt` and
`asr_metrics-phone1943-canonic-epoch-2.json` under the run's `greedy_search/`.
Checkpoint suffix changes when evaluating a later epoch; keep the same dataset
and decoding settings. The baseline is ASR only, with no conversation-history
or agent-fact annotations.

Validation: five CPU tests pass (new TSV tests plus existing ASR-cut tests),
including waveform/reference preservation, short evaluation items, isolated
transcript targets and invalid path/duration rejection. The full exported set
is reread and decoded for readiness separately from GPU evaluation.

## Completed epoch-2 result

The unchanged native evaluator completed all 1,943 records on the assigned
exp GPU3 and reused the existing HF export. The process exited successfully;
no subsequent GPU job was launched by this preparation task. Its log is
`phone-test-eval-v1/native-epoch2-eval.log`. The initial missing-Auden import
failure is preserved as `native-epoch2-eval-attempt1.log`; it preceded model
loading. The successful invocation used the native runtime documented above.

| Same reference set and normalizer | GigaAM baseline WER / CER | IFAO epoch-2 WER / CER |
| --- | ---: | ---: |
| Native IFAO | 20.3219% / 8.02161% | 43.5578% / 20.40270% |
| GigaAM normalization | 17.8653% / 7.50439% | 42.4142% / 20.09021% |

IFAO has 5,007 normalized word errors over the same 11,805 reference words,
618 exact transcripts and two empty outputs. Thirteen hypotheses exceed three
times their reference word count and contribute 1,852 errors; inspecting the
largest cases shows repeated short phrases. This diagnostic does not remove
any records or change the official score. Keep `max_new_tokens=256` for the
next checkpoint comparison; changing the cap would affect these failures.
No digit verbalization or reference substitution was applied.

`phone-test-eval-v1/native-epoch2-qc/` contains `metrics.json`, `cases.jsonl`,
copies of the untouched native outputs, hashes of the evaluator, manifest,
model configuration and predictions, and the exact generation settings.
Every native reference was rechecked against its original TSV row before
rescoring, and the independently recomputed native metric matches the
evaluator's result. The original source labels remain evaluation-only.
