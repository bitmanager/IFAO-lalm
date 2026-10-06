# Independent source QC and epoch-2 diagnostics

All runs completed on the allocated exp GPU1, UUID
`GPU-0f7a532e-ef28-779c-fc1f-87c7203f703a`. Immediately before each launch,
`nvidia-smi` showed 2 MiB and no compute processes on that GPU. Jobs ran
sequentially and exited successfully. The final check again showed 2 MiB and
no GPU1 process. Training on GPUs 0/2 and the demo on GPU3 were not changed.

## Clean-source GigaAM check

The unchanged upstream `train_utils/eval.py` and its TSV format/normalizer
were used with the existing `/model.ckpt` GigaAM RNNT checkpoint. The checkpoint
hash is `607e71d4d9fa87fadbdd779357fc3d1d064e1c273e7080cafd325ce474ba591e`.
No new decoding code, teacher labels or transcript replacements were used.
Both foreground and background isolated sources were evaluated, never the
mixtures as if a baseline recognizer supplied gold labels.

342 role exports produced 332 distinct WAV **container** hashes, but only
218 distinct decoded PCM sources (0.37515556 hours). Repeated FLOAT WAV
exports have PEAK-chunk timestamp differences in their headers despite
identical samples. Predictions agree across every repeated PCM source.
The original 332-record upstream run remains intact: normalized WER 3.6338%
(181 / 4,981 words). Deduplicating identical PCM gives **3.5052%**
(119 / 3,395), with **155 / 218** normalized exact records. Foreground-only:
98 / 2,561 = 3.8266%; background-only: 88 / 2,564 = 3.4321%.
Each role contains 171 distinct waveforms; shared sources across roles account
for the smaller union. No numeric verbalization or custom normalization was
added. Punctuation-sensitive upstream WER is retained in `metrics.json`, but
is not the acoustic-quality figure.

The adapter now checks decoded little-endian float32 sample hashes for split
leakage; file hashes remain for artifact integrity. A CPU reread confirmed
zero PCM overlap between the existing full train and validation exports.
This guard fix does not regenerate or change any evaluated WAV or manifest.
Seven tests pass, including rejection of equal PCM with unequal container
hashes and actual native mixer/export execution.

The complete artifacts are under
`/runs/dev-storage/ifao-data/context-overlap-v1/asr-qc-gigaam-v1/`:
`isolated-sources.tsv`, untouched upstream `predictions/`, `eval.log`,
`cases.jsonl`, `metrics.json`, `source-role-index.jsonl` and
`pcm-source-index.jsonl`. The latter two retain every pair/role mapping.

## Quarantine and proposed admission

`assets/context_overlap_source_quarantine.jsonl` identifies five source cut
IDs with suspected large omissions: missing name/date clauses, whole second
sentences or alternatives, and a contracted WAV/FLAC question. These affect
**six train pairs and one validation pair**. This is conservative quarantine
based on independent clean ASR, not a claim of human-confirmed TTS defects.
Listening is still pending; model error remains possible, especially around
technical terms. No source or reference has been deleted or rewritten.

`assets/context_overlap_clean_qc_mask.jsonl` is an independent stricter mask:
both isolated roles must match the GigaAM-normalized source text exactly.
It yields **83 train pairs** and **5 validation pairs**. No native epoch-2
score or prediction enters this decision. The complete fixed 12-pair
validation remains unchanged and is always reported first. Its five-pair
masked diagnostic is additional, not a replacement test set.

Proposed next admission: begin with those 83 train pairs only (332 audio
variants, 664 task views), subject to explicit admission and review of the
existing teacher answers. Exact ASR agreement is useful screening, not human
gold, and says nothing about factual teacher-answer quality. Keep the six
quarantined train pairs out; retain the other mismatches for review rather
than silently accepting or automatically relabeling them. All current cuts
and mask entries remain `training_eligible=false`; no train config changes.

## Native epoch-2 fixed validation

The existing HF export `/runs/ifao-balalaika-export/epoch-2` was reused by
unchanged `evaluate_qa.py`, bf16 greedy, `max_new_tokens=256`, batch duration
budget 40 seconds. ASR and answer YAMLs each evaluated all four conditions
and all 12 foregrounds. Both original task instructions and text histories
were supplied; last-assistant labels were stripped by the native evaluator.

| Condition | Full fixed 12: native-normalized ASR WER | Independent clean-QC mask, 5: WER |
| --- | ---: | ---: |
| Clean | 18.0791% | 14.4737% |
| +3 dB | 201.6949% | 44.7368% |
| 0 dB | 75.7062% | 64.4737% |
| −3 dB | 94.3503% | 88.1579% |

The +3 dB full score includes one output longer than three times its
reference; all errors remain in the official score. These conditions do not
establish a monotonic SNR curve on 12 examples. They expose selection and
generation failures; a first-voice instruction alone has not solved them.

Full answer strings, exact histories, target/background transcripts,
unchanged teacher references, ASR hypotheses and source-QC masks are saved
together in `native-epoch2/cases.jsonl` (48 condition/pair records). Native
raw outputs remain in `native-epoch2/greedy_search/`; summary metrics are
`native-epoch2/summary.json`. The native answer containment metric is zero
for every condition, but is unsuitable as factual answer accuracy here.
Examples from the actual outputs:

* Pair `0161`: history assigns game roles; the current utterance says Gosha
  is 30 minutes late and asks whether to wait. Clean ASR retains that fact,
  but the agent answers **“Гоша начинает без задержки.”** At +3 dB its answer
  changes the name to Borya and the delay to 13 minutes. At −3 dB it responds
  with unrelated map/building words from the background. Both isolated
  sources passed exact clean ASR; this case is in the five-pair mask.
* Pair `0164`: history contains podcast pieces of 12, 8 and 15 minutes.
  Clean, +3 dB and 0 dB answers all recover **35 minutes**, although their
  wording differs from the teacher reference. At −3 dB the answer instead
  asks what should be glued, following the unrelated background. This pair
  is outside the strict clean-source mask and remains in full validation.

These are individual text-based fact observations, not a scored fact rubric
or a human listening review. Existing teacher references themselves contain
questionable technical advice; matching them is not proof of correctness.

## Exact commands

Inside the existing exp container, source QC used
`PYTHONPATH=/deps:/review/vendor:/olddeps` (PyTorch 2.11, encoder fp16,
default `use_flash=false`), `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`:

```sh
python /tmp/foreground-gigaam-upstream/train_utils/eval.py \
  --eval_manifest /runs/dev-storage/ifao-data/context-overlap-v1/asr-qc-gigaam-v1/isolated-sources.tsv \
  --checkpoint /runs/dev-storage/ifao-data/context-overlap-v1/asr-qc-gigaam-v1/gigaam-sip.ckpt \
  --batch_size 16 --num_workers 2 --device cuda
```

That checkpoint path is a symlink to `/model.ckpt`, placing upstream outputs
in this QC tree rather than the shared `/model/preds.jsonl`. The source TSV
and upstream evaluator/utils/checkpoint hashes are retained in `metrics.json`.

Native evaluation instead used
`PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash`
(existing PyTorch 2.9 / matching FlashAttention), and also `MKL_NUM_THREADS=2`.
Both runtimes used `docker exec -u 0:26403` and the GPU1 UUID above:

```sh
python /ifao-context-review/lalm/evaluate_qa.py \
  exp_dir=/runs/dev-storage/ifao-data/context-overlap-v1/native-epoch2 \
  checkpoint.model_dir=/runs/ifao-balalaika-export/epoch-2 checkpoint.epoch=2 \
  data.test_data_config=/runs/dev-storage/ifao-data/context-overlap-v1/eval-asr.yaml \
  data.max_duration=40 data.num_workers=2 +asr=true dtype=bf16 max_new_tokens=256
```

The subsequent answer command is identical except `eval-answer.yaml` and
omitting `+asr=true`. Logs are `native-epoch2-asr.log` and
`native-epoch2-answer.log` at the dataset root. No other GPU run was launched.
