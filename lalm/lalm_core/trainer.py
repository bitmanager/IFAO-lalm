from __future__ import annotations

import copy
import logging
import math

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class LALMTrainer(BaseTrainer):

    def __init__(self, cfg, model, data_module, rank=0, local_rank=0, world_size=1):
        t = cfg.trainer
        self.response_kl = bool(t.get("response_kl", False))
        self.response_kl_temperature = float(t.get("response_kl_temperature", 2.))
        self.answer_loss_weight = t.get("answer_loss_weight", None)
        if self.response_kl != bool(cfg.data.get("response_kl", False)):
            raise ValueError("Trainer and data response_kl modes must match")
        if self.response_kl:
            if self.answer_loss_weight is not None:
                raise ValueError("Response KL requires matching data mode and no answer CE weight")
            if not math.isfinite(self.response_kl_temperature) or self.response_kl_temperature <= 0:
                raise ValueError("Response KL temperature must be positive and finite")
        if self.answer_loss_weight is not None:
            self.answer_loss_weight = float(self.answer_loss_weight)
            if not math.isfinite(self.answer_loss_weight) or self.answer_loss_weight < 0:
                raise ValueError("answer_loss_weight must be finite and nonnegative")
        self._grad_accum_steps: int = int(getattr(t, "grad_accum_steps", 1))
        self._max_grad_norm: float = float(getattr(t, "max_grad_norm", 1.0))
        self._ema_decay: float = float(getattr(t, "ema_decay", 0.9999))
        self._accum_step: int = 0
        super().__init__(cfg, model, data_module, rank, local_rank, world_size)

    def setup_model(self, model: nn.Module):
        """float32 EMA on rank-0 (not float64), find_unused_parameters=False."""
        if self.response_kl:
            model.validate_response_kl_freeze()
        if self.rank == 0:
            model_avg = copy.deepcopy(model).to(torch.float32).to("cpu")
        else:
            model_avg = None

        model = model.to(self.device)

        # GradScaler (fp16) requires fp32 params so that grads are fp32.                                                                                
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:                                                                                            
                p.data = p.data.float() 
        

        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.trainer.get(
                    "find_unused_parameters", False
                ),
            )

        num_param = sum(p.numel() for p in model.parameters()) / 1e6
        num_trainable = (
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
        logging.info(
            f"Parameters: {num_param:.2f}M total, {num_trainable:.2f}M trainable"
        )
        return model, model_avg

    def _maybe_update_model_average(self):
        """EMA over trainable params only — skips frozen weights to avoid copying
        the entire model (e.g. 7B LLM backbone) from GPU to CPU every N steps."""
        if (
            self.rank != 0
            or not self.cfg.trainer.get("use_averaged_model", False)
            or self.global_step == 0
            or self.global_step % self.cfg.trainer.average_period != 0
        ):
            return

        model_cur = self.model.module if isinstance(self.model, DDP) else self.model
        decay = self._ema_decay
        with torch.no_grad():
            for (_, avg_p), (_, cur_p) in zip(
                self.model_avg.named_parameters(),
                model_cur.named_parameters(),
            ):
                if not cur_p.requires_grad:
                    continue
                avg_p.data.mul_(decay).add_(cur_p.data.float().cpu(), alpha=1.0 - decay)

    def _forward_backward_optimize(self, batch: dict):
        """zero_grad at start of accumulation window, grad accumulation, grad clipping."""
        amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else None
        )

        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            loss, batch_metrics = self._forward_one_batch(batch, is_training=True)

        scaled_loss = loss / self._grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

        self._accum_step += 1

        if self._accum_step >= self._grad_accum_steps:
            self._accum_step = 0

            self.scaler.unscale_(self.optimizer)

            if self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.grad is not None),
                    self._max_grad_norm,
                )

            self.scheduler.step_batch(self.global_step)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss, batch_metrics

    def _forward_one_batch(self, batch, is_training=True):
        device = self.device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        audio_features = batch["features"].to(device, non_blocking=True)
        feature_lens = batch["feature_lens"].to(device=device, dtype=torch.long, non_blocking=True)

        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        audio_param = next(model_ref.audio_tower.parameters(), None)
        if model_ref.config.audio_config.model_type != "gigaam" and audio_param is not None and audio_features.dtype != audio_param.dtype:
            audio_features = audio_features.to(dtype=audio_param.dtype)

        amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else None
        )

        with torch.set_grad_enabled(is_training), torch.amp.autocast(
            "cuda", enabled=amp_dtype is not None, dtype=amp_dtype
        ):
            # Validation may have unequal batch counts across ranks. Avoid DDP
            # forward buffer broadcasts; reduce task statistics once per split.
            task_normalized = self.answer_loss_weight is not None or self.response_kl
            forward_model = model_ref if task_normalized and not is_training else self.model
            teacher_inputs = None
            if self.response_kl:
                teacher_inputs = {k: v.to(device, non_blocking=True)
                                  for k, v in batch["teacher_inputs"].items()}
            outputs = forward_model(
                input_ids=input_ids,
                audio_features=audio_features,
                feature_lens=feature_lens,
                attention_mask=attention_mask,
                labels=labels,
                asr_mask=batch["asr_mask"].to(device, non_blocking=True),
                answer_loss_weight=self.answer_loss_weight,
                sync_task_counts=is_training and task_normalized,
                response_kl=self.response_kl,
                response_kl_temperature=self.response_kl_temperature,
                teacher_inputs=teacher_inputs,
            )
            loss = outputs.loss
            logits = outputs.logits
            packed_labels = getattr(outputs, "packed_labels", labels)

        info = MetricsTracker()
        B = int(input_ids.size(0))
        info.set_value("samples", B, normalization="sum")
        info.set_value(
            "tokens", int((attention_mask > 0).sum().item()), normalization="sum"
        )
        info.set_value(
            "loss", float(loss.detach().cpu().item()), normalization="sample_avg"
        )
        info.set_value(
            "acc",
            float(self._token_accuracy(logits, packed_labels).detach().cpu().item()),
            normalization="sample_avg",
        )
        if self.answer_loss_weight is not None:
            for i, task in enumerate(("asr", "answer")):
                info.set_value(f"{task}_nll", outputs.task_nll_sums[i].item(), "sum")
                info.set_value(f"{task}_target_tokens", outputs.task_token_counts[i].item(), "sum")
        if self.response_kl:
            info.set_value("response_kl_sum", outputs.response_kl_sum.item(), "sum")
            info.set_value("response_target_tokens", outputs.response_target_tokens.item(), "sum")
        return loss, info

    def validate(self, epoch):
        if self.answer_loss_weight is None and not self.response_kl:
            return super().validate(epoch)
        # Same native validation iteration/reduction; task numerators and counts
        # are summed before division, rather than averaging batch/task means.
        self.model.eval()
        with torch.no_grad():
            for name, loader in zip(self.data_module.valid_names, self.data_module.valid_dls):
                stats = dict(asr_nll=0., answer_nll=0., asr_target_tokens=0., answer_target_tokens=0.,
                    samples=0., tokens=0., sample_accuracy_sum=0.)
                if self.response_kl:
                    stats = dict(response_kl_sum=0., response_target_tokens=0.,
                                 samples=0., tokens=0., sample_accuracy_sum=0.)
                for batch in loader:
                    _, metrics = self._forward_one_batch(batch, is_training=False)
                    for key in stats:
                        if key != "sample_accuracy_sum":
                            stats[key] += metrics._values[key]
                    stats["sample_accuracy_sum"] += metrics._values["acc"] * metrics._values["samples"]
                total = MetricsTracker()
                for key, value in stats.items():
                    total.set_value(key, value, "sum")
                if self.world_size > 1:
                    total.reduce(self.device)
                if self.response_kl:
                    mean = total._values["response_kl_sum"] / max(total._values["response_target_tokens"], 1)
                    values = dict(response_kl=mean, loss=mean)
                else:
                    values = task_validation_values(total._values, self.answer_loss_weight)
                values["acc"] = total._values["sample_accuracy_sum"] / max(total._values["samples"], 1)
                for key, value in values.items():
                    total.set_value(key, value)
                if self.rank == 0:
                    logging.info(f"Epoch {epoch}, global step {self.global_step}, validation {name}: {total}")
                    if self.tb_writer is not None:
                        total.write_summary(self.tb_writer, f"train/valid_{name}_", self.global_step)
        self.model.train()

    @staticmethod
    def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid = shift_labels != -100
        n = valid.sum()
        if n == 0:
            return logits.new_zeros(())
        preds = shift_logits.argmax(-1)[valid]
        correct = (preds == shift_labels[valid]).sum()
        return correct.float() / n


def task_validation_values(stats, answer_weight):
    asr = stats["asr_nll"] / max(stats["asr_target_tokens"], 1)
    answer = stats["answer_nll"] / max(stats["answer_target_tokens"], 1)
    return dict(asr_ce=asr, answer_ce=answer, loss=asr + answer_weight * answer)
