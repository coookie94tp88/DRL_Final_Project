import numpy as np
import torch

from doorl.models.transformer_policy import _beta_params
from doorl.ppo_utils import normalize_advantages, ppo_surrogate_loss, safe_optimizer_step


def test_normalize_advantages_clip():
    adv = np.array([100.0, -100.0, 0.0])
    out = normalize_advantages(adv, clip=5.0)
    assert np.abs(out).max() <= 5.0 + 1e-6


def test_beta_params_nan_input():
    raw = torch.tensor([[float("nan"), float("inf")]])
    a, b = _beta_params(raw)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert a.max() <= 100.0


def test_safe_optimizer_step_skips_bad_loss():
    m = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.tensor(float("nan"))
    ok, reason = safe_optimizer_step(opt, [m], loss, 0.5)
    assert not ok
    assert reason == "non_finite_loss"
