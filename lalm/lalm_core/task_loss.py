"""Opt-in task-normalized causal CE; inference and default pooled CE are unchanged."""
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def separate_task_ce(logits, labels, asr_positions, answer_weight, sync_counts=False):
    if not math.isfinite(answer_weight) or answer_weight < 0:
        raise ValueError("answer_loss_weight must be finite and nonnegative")
    targets = labels[..., 1:].reshape(-1)
    positions = asr_positions[1:].reshape(-1)
    valid = targets != -100
    nll = F.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.size(-1)),
                         targets, ignore_index=-100, reduction="none")
    masks = torch.stack((valid & positions, valid & ~positions))
    sums = (masks * nll).sum(dim=1)
    counts = masks.sum(dim=1)
    denominators = counts.detach().clone()
    world = 1
    if sync_counts and dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominators)
        world = dist.get_world_size()
    # DDP averages gradients: compensate to get each task's global token mean,
    # including a rank with zero examples of that task. Empty tasks contribute 0.
    means = sums * world / denominators.clamp_min(1)
    loss = means[0] + answer_weight * means[1]
    return loss, sums.detach(), counts.detach()


def response_kl_loss(student_logits, teacher_logits, student_labels, teacher_labels,
                     temperature=2., sync_counts=False):
    """KL(teacher || student), FP32, token mean; no T**2, as in BLSP-Emo.

    Reference: https://github.com/cwang621/blsp-emo/blob/e0f45042d9c2f89cbceab7e1d7eebc33ce0459f8/src/modeling_blsp2.py
    Adaptations: shifted native response masks and DDP global token mean.

    Inputs are packed independently after per-example target identity checks.
    Shifted labels select predictions of first answer token through EOS, not the
    distribution after EOS. Packed segment starts must be masked by the caller.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Response KL temperature must be positive and finite")
    smask = student_labels[..., 1:] != -100
    tmask = teacher_labels[..., 1:] != -100
    if not torch.equal(student_labels[..., 1:][smask], teacher_labels[..., 1:][tmask]):
        raise ValueError("Response KL packed targets differ")
    student = student_logits[..., :-1, :][smask].float() / temperature
    teacher = teacher_logits[..., :-1, :][tmask].detach().float() / temperature
    # Keep the entire softmax/KL in FP32, even under the trainer's autocast.
    with torch.autocast(device_type=student.device.type, enabled=False):
        numerator = F.kl_div(F.log_softmax(student, dim=-1),
                             F.softmax(teacher, dim=-1), reduction="sum")
    count = smask.sum()
    denominator = count.detach().clone()
    world = 1
    if sync_counts and dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator)
        world = dist.get_world_size()
    return numerator * world / denominator.clamp_min(1), numerator.detach(), count.detach()
