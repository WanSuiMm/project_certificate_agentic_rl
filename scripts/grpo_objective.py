"""Token-level clipped GRPO loss with a frozen reference-policy KL term."""

from __future__ import annotations

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, *, group_size: int, eps: float = 1e-6) -> Tensor:
    if rewards.ndim != 1 or rewards.numel() % group_size or group_size < 2:
        raise ValueError("rewards must be flat and divisible into groups")
    grouped = rewards.float().reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    normalized = torch.where(std > eps, (grouped - mean) / std.clamp_min(eps), torch.zeros_like(grouped))
    return normalized.flatten()


def token_log_probs(model: torch.nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Log probability for target token at each position 1..L-1."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shifted = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return torch.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def clipped_grpo_loss(
    new_logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-completion token mean, then batch mean; old/ref tensors are fixed."""
    if not (new_logp.shape == old_logp.shape == ref_logp.shape == completion_mask.shape):
        raise ValueError("token tensors and mask must have identical shapes")
    if advantages.shape != (new_logp.shape[0],):
        raise ValueError("one group-relative advantage is required per completion")
    if not 0 < clip_epsilon < 1 or kl_beta < 0:
        raise ValueError("invalid clipping or KL coefficient")
    mask = completion_mask.to(dtype=new_logp.dtype)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("empty completion")
    log_ratio = (new_logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    advantage = advantages.to(new_logp.dtype).unsqueeze(1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.minimum(unclipped, clipped)
    # Schulman nonnegative reverse-KL estimator used in GRPO implementations.
    ref_minus_new = (ref_logp - new_logp).clamp(-20.0, 20.0)
    kl = ref_minus_new.exp() - ref_minus_new - 1.0
    per_completion = ((policy_loss + kl_beta * kl) * mask).sum(dim=1) / mask.sum(dim=1)
    loss = per_completion.mean()
    return loss, {
        "mean_kl": ((kl * mask).sum() / mask.sum()).detach(),
        "clip_fraction": ((((ratio - 1.0).abs() > clip_epsilon).to(mask.dtype) * mask).sum() / mask.sum()).detach(),
    }
