"""BC pretrain raises follow-public rate vs random init."""

from doorl.bc_pretrain import collect_bc_samples, train_bc
from doorl.config_utils import load_config
from doorl.eval import run_episodes
from doorl.baselines import TruthfulHost
from doorl.train import _build_models


def test_bc_pretrain_improves_follow_public():
    cfg = load_config(
        "config/default.yaml",
        overrides=[
            "env.num_players=4",
            "env.max_rounds=20",
            "model.d_model=64",
            "model.n_layers=1",
            "model.nhead=4",
        ],
    )
    player, _, _ = _build_models(cfg)
    host_bl = TruthfulHost(4, seed=0)
    before = run_episodes(
        cfg, player_policy=player, host_baseline=host_bl, episodes=30, seed=0
    )

    samples = collect_bc_samples(cfg, episodes=80, seed=0)
    assert len(samples) > 200
    train_bc(player, samples, epochs=15, batch_size=128, lr=3e-4, device="cpu")

    after = run_episodes(
        cfg, player_policy=player, host_baseline=host_bl, episodes=30, seed=0
    )
    assert after.p_follow_public > before.p_follow_public + 0.08
