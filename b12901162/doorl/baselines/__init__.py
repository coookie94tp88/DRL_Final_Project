"""Scripted policies used as evaluation baselines.

Each baseline implements ``act_host(env, obs) -> action_dict`` and/or
``act_player(env, obs, i) -> action_dict``. They are deliberately stateless
and rng-driven so they compose cleanly inside ``doorl.eval``.
"""

from doorl.baselines.random_host import RandomHost
from doorl.baselines.truthful_host import TruthfulHost
from doorl.baselines.noisy_truthful_host import NoisyTruthfulHost
from doorl.baselines.greedy_players import GreedyPlayer
from doorl.baselines.no_bribe_players import NoBribePlayer

__all__ = [
    "RandomHost",
    "TruthfulHost",
    "NoisyTruthfulHost",
    "GreedyPlayer",
    "NoBribePlayer",
]
