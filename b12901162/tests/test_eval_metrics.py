"""Eval metrics must use last_settlement, not post-step env._round_* fields."""

from doorl.baselines import GreedyPlayer, TruthfulHost
from doorl.config_utils import load_config
from doorl.eval import run_episodes


def test_truthful_host_mi_near_max_bits():
    """Public/private signals track true door → I(signal; truth) ≈ 2 bits (4 doors)."""
    cfg = load_config(
        "config/default.yaml",
        overrides=["env.num_players=4", "env.max_rounds=50"],
    )
    host = TruthfulHost(num_players=4, seed=0)
    summary = run_episodes(
        cfg,
        host_baseline=host,
        episodes=80,
        seed=0,
    )
    assert summary.rounds_collected > 100
    assert summary.mi_public_truth > 1.5, summary.mi_public_truth
    assert summary.mi_private_truth > 1.0, summary.mi_private_truth


def test_truthful_host_with_greedy_players_follows_signals():
    cfg = load_config(
        "config/default.yaml",
        overrides=["env.num_players=4", "env.max_rounds=30"],
    )
    summary = run_episodes(
        cfg,
        host_baseline=TruthfulHost(num_players=4, seed=1),
        player_baseline=GreedyPlayer(),
        episodes=40,
        seed=1,
    )
    assert summary.p_follow_public > 0.7, summary.p_follow_public
    assert summary.p_follow_private > 0.7, summary.p_follow_private
    assert summary.mi_public_truth > 1.5
