# Linked ASR and answer supervision (opt-in)

`gigaam_paired_tasks` retains the native model, trainer loop, sampler and inference.
It changes the training objective to `CE_ASR + answer_loss_weight * CE_answer`.
Each CE is normalized by that task's supervised tokens, including EOS. The
default configuration and existing manifests still use the original pooled CE.

Prepare **base** MonoCuts whose single supervision has the current transcript in
`text` and an explicitly labelled response in `custom.answer`. Existing system,
history, audio and source IDs are retained. No response is generated or inferred
from the transcript. A missing/blank label omits its task; a cut with neither
label is rejected. Already task-adapted cuts are rejected.

```bash
python prepare_paired_tasks.py --input-manifest BASE.jsonl.gz \
  --output-manifest LINKED.jsonl.gz --model-dir NATIVE_HF_EXPORT
# Use LINKED.jsonl.gz in the existing train-data YAML. Existing single-task
# manifests can also be mixed in with the native data selectors.
torchrun --standalone --nproc_per_node=2 train.py \
  --config-name gigaam_paired_tasks trainer.answer_loss_weight=2.0
```

The usual `IFAO_RUN_DIR`, `IFAO_MODEL_PATH`, `IFAO_TRAIN_CONFIG` and
`IFAO_VALID_CONFIG` apply. This configuration does not change the inherited
freeze policy, optimizer, learning rate or checkpoint handling. These example
commands do not launch a job automatically or select a data recipe.

A manifest unit contains one or two prepared task views. The existing token
sampler budgets **both** text/audio sequences and keeps the unit on one rank in
one batch. The dataset expands the views after sampling and cut transforms;
the two sequences receive the same transformed audio/history, different task
prompts and their own original targets. They remain causally isolated when
packed. Audio is encoded for each view: this deliberately reuses the existing
forward path, and does not promise free extra supervision.

The ASR sequence uses the intermediate ASR readout; the answer uses the native
final LM head. Gradients follow the configured freeze policy. Head selection,
autoregressive generation and inference prompts are unchanged.

During DDP training each task denominator is summed across ranks; local loss
is scaled to compensate for DDP's gradient averaging. A missing task/rank or
entirely masked labels contribute finite zero, without inventing a target.
The empty ASR route retains a zero gradient connection to the ASR readout, so
`find_unused_parameters=False` remains usable when that head is trainable.
Native gradient accumulation still averages the microbatch objectives.

Opt-in validation sums each task's NLL and token count before reporting
`asr_ce`, `answer_ce` and their weighted sum. Counts are reported, so an absent
task's zero is distinguishable from a measured zero CE. The small validation
override is needed because the native metric tracker has only one shared
sample/frame denominator. It calls the unwrapped model in evaluation mode to
avoid DDP forward buffer collectives when ranks have unequal validation lengths;
statistics are reduced once per split. Default validation is unchanged.

Tests cover original labels/history/EOS, missing targets, atomic two-rank
sampling, padding/packed boundaries, frozen-backbone projector gradients,
lambda scaling, all-ignored loss, unchanged default CE, and CPU/Gloo task/rank
imbalance including an empty validation rank. The optional real-processor test
uses an existing local native export and never downloads or starts a GPU job.
