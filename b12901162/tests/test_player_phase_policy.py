"""Player policy must gate actions/logps by env phase (public only in BET obs)."""

import numpy as np
import torch

from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE
from doorl.models.transformer_policy import PlayerPolicy, PlayerPolicyConfig


def _policy():
    cfg = PlayerPolicyConfig(obs_dim=64, num_players=4, d_model=32, n_layers=1, nhead=4)
    return PlayerPolicy(cfg)


def test_act_bribe_phase_zeroes_door_logp_in_total():
    pol = _policy()
    obs = torch.randn(2, 64)
    idx = torch.tensor([0, 1], dtype=torch.long)
    act, info = pol.act(obs, idx, phase=PHASE_BRIBE, deterministic=True)
    assert np.asarray(act["bribe_pct"]).shape[0] == 2
    # door/bet are placeholders in bribe phase
    assert float(info["logp_bribe"].abs().sum()) > 0
    assert float(info["logp"].abs().sum()) == float(info["logp_bribe"].abs().sum())


def test_act_bet_phase_uses_door_and_bet_logp():
    pol = _policy()
    obs = torch.randn(2, 64)
    idx = torch.tensor([0, 1], dtype=torch.long)
    _, info = pol.act(obs, idx, phase=PHASE_BET, deterministic=True)
    total = info["logp_door"] + info["logp_bet"]
    assert torch.allclose(info["logp"], total)


def test_evaluate_respects_phase_mask():
    pol = _policy()
    obs = torch.randn(3, 64)
    idx = torch.zeros(3, dtype=torch.long)
    actions = {
        "bribe_pct": torch.full((3,), 0.1),
        "door": torch.tensor([0, 1, 2]),
        "bet_pct": torch.full((3,), 0.2),
    }
    phases = torch.tensor([PHASE_BRIBE, PHASE_BET, PHASE_BRIBE], dtype=torch.long)
    ev = pol.evaluate(obs, idx, actions, phases=phases)
    expected = torch.zeros(3)
    expected[0] = ev["logp_bribe"][0]
    expected[1] = ev["logp_door"][1] + ev["logp_bet"][1]
    expected[2] = ev["logp_bribe"][2]
    assert torch.allclose(ev["logp"], expected, atol=1e-5)
