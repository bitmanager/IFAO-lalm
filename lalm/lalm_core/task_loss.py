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
