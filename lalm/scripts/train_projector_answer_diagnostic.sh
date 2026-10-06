#!/usr/bin/env bash
# Root review required before invocation. Fresh projector diagnostic, bounded updates.
set -euo pipefail
export IFAO_ANSWER_SELECTOR=original-script
exit_path=/runs/ifao-projector-answer-diagnostic.exit
export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-projector-answer-only-diagnostic-v1
updates=250
valid_interval=50000
if [[ $# -gt 0 ]]; then
  [[ $# -eq 1 ]] || { echo 'Expected no arguments, --self-teacher or --joint-self-teacher' >&2; exit 2; }
  case "$1" in
    --self-teacher)
      export IFAO_ANSWER_SELECTOR=self-teacher
      export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-projector-self-teacher-diagnostic-v1
      exit_path=/runs/ifao-projector-self-teacher-diagnostic.exit ;;
    --joint-self-teacher)
      export IFAO_ANSWER_SELECTOR=joint-self-teacher
      export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-projector-joint-selfteacher-v1
      exit_path=/runs/ifao-projector-joint-selfteacher.exit
      updates=500
      valid_interval=501 ;;
    *) echo 'Expected no arguments, --self-teacher or --joint-self-teacher' >&2; exit 2 ;;
  esac
fi
export IFAO_DIAGNOSTIC_UPDATES="$updates"
trap 'status=$?; printf "%s\n" "$status" > "$exit_path"' EXIT
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
export IFAO_MODEL_PATH=/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$IFAO_RUN_DIR/validation.yaml"
cd /ifao-context-review/lalm
python - <<'PY'
import json,os
from pathlib import Path
from scripts.prepare_projector_semantic_v3 import sha256
out=Path(os.environ['IFAO_RUN_DIR']);r=json.loads((out/'readiness.json').read_text())
selector=os.environ['IFAO_ANSWER_SELECTOR'];expected={'original-script':2538,'self-teacher':2520,'joint-self-teacher':21588}[selector]
updates=int(os.environ['IFAO_DIAGNOSTIC_UPDATES'])
assert r.get('selector','original-script')==selector
assert r['status']=='ready_for_review_not_launched' and r['cuts']==expected
assert r['tasks']==({'asr':19068,'answer':2520} if selector=='joint-self-teacher' else {'answer':expected})
assert r['updates']==updates
if selector in ('self-teacher','joint-self-teacher'):
    assert r['unique_original_turns']==840 and r['source_provenance']['quality']=='same_frozen_qwen_self_generated_not_gold'
if selector=='joint-self-teacher':
    assert r['retained_asr_exact_rows']==19068 and r['retained_asr_ids_sha256']=='56666aa350742e9608629fb49ad5b1e606fd5defbe1ddf958a9af4fb22fd84c9'
    assert r['native_cpu_batch']['asr']==12 and r['native_cpu_batch']['answer']==4
assert r['trainable_parameters']==10490880 and r['model_dir']==os.environ['IFAO_MODEL_PATH']
assert r['native_cpu_batch']['exact_final_target_and_EOS'] and min(r['finite_sampler']['batches_per_rank'].values())>=updates
assert not (out/'hf').exists() and not list(out.glob('*.pt')), 'Fresh run/optimizer only'
checks={**r['input_sha256'],**r['model_file_sha256'],**r['heldout_manifest_sha256'],
        r['manifest']:r['manifest_sha256'],os.environ['IFAO_TRAIN_CONFIG']:r['train_config_sha256'],
        os.environ['IFAO_VALID_CONFIG']:r['validation_config_sha256']}
for path,digest in checks.items(): assert sha256(path)==digest,path
print(f'Preflight passed: {expected} {selector} task views, fresh iter26000, {updates} updates, projector only.',flush=True)
PY
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=1 trainer.start_batch=0 \
  trainer.num_epochs=1 trainer.num_steps=$((updates - 1)) trainer.grad_accum_steps=1 \
  'trainer.frozen_modules=[audio_tower,language_model,asr_head,asr_norm]' \
  trainer.optimizer.lr=3e-6 trainer.optimizer.weight_decay=0.0 trainer.scheduler.eta_min=3e-6 \
  trainer.mixed_precision=bf16 trainer.valid_interval="$valid_interval" trainer.save_every_n=1 trainer.keep_last_k=2 \
  data.sampler.max_tokens=2000
