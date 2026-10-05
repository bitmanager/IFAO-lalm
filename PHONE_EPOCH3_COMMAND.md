# Frozen phone evaluation: next checkpoint

The completed epoch-2 command was run on exp-1 with the existing container
and runtime below. The full audit and identical-normalizer baseline results
are in [PR5](https://github.com/bitmanager/IFAO-lalm/pull/5).

```sh
sudo docker exec -u 0:26403 \
  -e CUDA_VISIBLE_DEVICES=GPU-35138370-770d-67f1-30d5-871db809cf99 \
  -e PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  nemo-asr-ru-20261005 \
  python /ifao-context-review/lalm/evaluate_qa.py \
  exp_dir=/data/ifao-runs/asr-balalaika-v1 \
  checkpoint.model_dir=/runs/ifao-balalaika-export/epoch-2 checkpoint.epoch=2 \
  data.test_data_config=/runs/dev-storage/ifao-data/phone-test-eval-v1/native/eval.yaml \
  data.max_duration=40 data.num_workers=2 +asr=true dtype=bf16 max_new_tokens=256
```

The scheduled next run root is
`/runs/dev-storage/ifao-data/runs/asr-phone-youtube-v1`. Once its epoch-3
checkpoint is complete and the root agent allocates GPU3, the corresponding
command is below. **This preparation task has not launched it.** Native
`prepare_model_dir` reuses an existing HF export with weights, otherwise it
exports the native checkpoint using the run's existing `hf/` metadata.
Do not mistake that metadata-only `hf/` directory for an epoch-3 export.

```sh
sudo docker exec -u 0:26403 \
  -e CUDA_VISIBLE_DEVICES=GPU-35138370-770d-67f1-30d5-871db809cf99 \
  -e PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  nemo-asr-ru-20261005 \
  python /ifao-context-review/lalm/evaluate_qa.py \
  exp_dir=/runs/dev-storage/ifao-data/runs/asr-phone-youtube-v1 \
  checkpoint.model_dir=/runs/dev-storage/ifao-data/runs/asr-phone-youtube-v1/export/epoch-3 \
  checkpoint.epoch=3 \
  data.test_data_config=/runs/dev-storage/ifao-data/phone-test-eval-v1/native/eval.yaml \
  data.max_duration=40 data.num_workers=2 +asr=true dtype=bf16 max_new_tokens=256
```

Output names in the new run's `greedy_search/` are
`qa_results-phone1943-canonic-epoch-3.txt` and
`asr_metrics-phone1943-canonic-epoch-3.json`. Keep the frozen 1,943 references,
manifest and generation settings unchanged. Compare native WER with the
same-normalizer GigaAM **20.3219%** and epoch-2 **43.5578%**. Rescoring the
same predictions with the original GigaAM normalizer gives **17.8653%** and
**42.4142%**, respectively; those are a separate comparison, not reference
substitutions. Dataset references are not independently certified human gold.
