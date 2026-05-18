"""PPO/HAPPO stability helpers (NaN guards, clipped advantages, safe steps)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def normalize_advantages(adv: np.ndarray, clip: float = 5.0) -> np.ndarray:
    adv = np.asarray(adv, dtype=np.float64)
    if clip > 0:
        adv = np.clip(adv, -clip, clip)
    std = float(adv.std())
    if std < 1e-8:
        return adv - adv.mean()
    return (adv - adv.mean()) / (std + 1e-8)


def ppo_surrogate_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    clip_range: float,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    log_ratio = torch.clamp(new_logp - old_logp, -log_ratio_clip, log_ratio_clip)
    ratio = log_ratio.exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    return -torch.min(surr1, surr2).mean()


def has_nonfinite_params(modules: Sequence[nn.Module]) -> bool:
    for module in modules:
        for p in module.parameters():
            if not torch.isfinite(p.data).all():
                return True
    return False


def safe_optimizer_step(
    optimizer: torch.optim.Optimizer,
    modules: Sequence[nn.Module],
    loss: torch.Tensor,
    grad_clip: float,
) -> Tuple[bool, str]:
    """Backprop + step; skip and zero grad if loss/grad/weights are non-finite."""
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False, "non_finite_loss"

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    params: List[torch.nn.Parameter] = []
    for module in modules:
        params.extend(module.parameters())

    if not params:
        optimizer.step()
        return True, "ok"

    grad_norm = nn.utils.clip_grad_norm_(params, grad_clip)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return False, "non_finite_grad"

    optimizer.step()

    if has_nonfinite_params(modules):
        optimizer.zero_grad(set_to_none=True)
        return False, "non_finite_params"

    return True, "ok"
