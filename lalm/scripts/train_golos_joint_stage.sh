#!/usr/bin/env bash
# Invoke only after parent review of full preparation/readiness. No preparation or resume here.
set -euo pipefail
export IFAO_RUN_DIR=/runs/dev-storage/ifao-data/runs/asr-projector-golos-joint-v1
export IFAO_MODEL_PATH=/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000
export IFAO_TRAIN_CONFIG="$IFAO_RUN_DIR/train.yaml"
export IFAO_VALID_CONFIG="$IFAO_RUN_DIR/validation.yaml"
export PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash
export OMP_NUM_THREADS=4
cd /ifao-context-review/lalm
updates=$(python - <<'PY'
import json,os
from pathlib import Path
from scripts import prepare_golos_joint_stage as preparation
from scripts.prepare_golos_joint_stage import EXIT,FROZEN,GOLOS,REPLAY_SHA,TEST_GATE,check_test_overlap,sha256
p=Path(os.environ['IFAO_RUN_DIR']);r=json.loads((p/'readiness.json').read_text())
assert EXIT.read_text().strip() == '0'
assert r['status']=='prepared_requires_parent_review_not_launched'
assert r['model_dir']==os.environ['IFAO_MODEL_PATH'] and r['frozen_modules']==FROZEN
assert r['trainable_parameters']==10490880 and r['replay_manifest_sha256']==REPLAY_SHA
assert r['native_cpu_batch']['exact_final_target_and_EOS']
assert sha256(preparation.__file__)==r['preparation_script_sha256']
assert r['unchanged_replay_rows_verified']==21588
s=r['finite_sampler'];n=s['updates']
assert s['batches_per_rank']=={'0':n,'1':n} and s['num_steps']==n-1
assert s['full_native_view_id_coverage'] and 0<s['N100']<=n and s['unique_golos_pcm_hours']>=100
assert not (p/'hf').exists() and not list(p.glob('*.pt')), 'Fresh run only; no sampler resume'
checks={**r['input_sha256'],r['manifest']:r['manifest_sha256'],
    os.environ['IFAO_TRAIN_CONFIG']:r['train_config_sha256'],
    os.environ['IFAO_VALID_CONFIG']:r['validation_config_sha256']}
gate=json.loads(TEST_GATE.read_text())  # Missing/pending audit blocks launch.
checks.update(check_test_overlap(gate,checks[str(GOLOS/'pilot-asr.jsonl.gz')],
    checks[str(GOLOS/'source-index.jsonl')]))
for path,digest in checks.items(): assert sha256(path)==digest,path
(p/'launch-gates.json').write_text(json.dumps(dict(official_test_overlap=gate,
    audit_sha256=sha256(TEST_GATE)),indent=2)+'\n')
print(n)
PY
)
trap 'status=$?; printf "%s\n" "$status" > /runs/ifao-golos-joint-stage.exit' EXIT
CUDA_VISIBLE_DEVICES=0,2 python -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_context trainer.start_epoch=1 trainer.start_batch=0 \
  trainer.num_epochs=1 trainer.num_steps=$((updates - 1)) trainer.grad_accum_steps=1 \
  'trainer.frozen_modules=[audio_tower,language_model,asr_head,asr_norm]' \
  trainer.optimizer.lr=3e-6 trainer.optimizer.weight_decay=0.0 trainer.scheduler.eta_min=3e-6 \
  trainer.mixed_precision=bf16 trainer.valid_interval=500 trainer.save_every_n=1 trainer.keep_last_k=2 \
  data.sampler.max_tokens=2000 data.use_infinite_dataset=false
