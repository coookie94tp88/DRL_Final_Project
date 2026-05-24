from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Phase(IntEnum):
    """One round: host proposes, then players vote (without seeing the proposal)."""
    PROPOSE = 0
    VOTE = 1


class Proposal(IntEnum):
    A = 0
    B = 1


class Vote(IntEnum):
    REJECT = 0
    ACCEPT = 1


@dataclass(frozen=True)
class ProposalVoteConfig:
    """Host–player proposal voting game."""

    num_players: int = 10
    max_rounds: int = 500
    history_window: int = 50

    @property
    def n(self) -> int:
        """Group size: players + host (used in proposal-A payoffs)."""
        return self.num_players + 1

    @property
    def current_player_dim(self) -> int:
        # [phase_vote_active, last_passed, last_own_vote, last_reward]
        return 4

    @property
    def hist_player_dim(self) -> int:
        # [own_vote, passed, reward, frac_accept, host_reward]
        return 5

    @property
    def current_host_dim(self) -> int:
        # [cumulative_host_reward, proposal_onehot_A, proposal_onehot_B, round_frac]
        return 4

    @property
    def host_player_state_dim(self) -> int:
        # [last_vote] — visible to host only during/after vote
        return 1

    @property
    def hist_host_dim(self) -> int:
        # [proposal, passed, host_reward, frac_accept] + per-player votes in players hist
        return 4


class ProposalVoteEnv(gym.Env):
    """Hidden-proposal majority vote game.

    Rules
    -----
    * Host chooses proposal **A** or **B** (players never observe which).
    * Only players vote **accept** (1) or **reject** (0).
    * Majority among players decides; **ties count as accept**.
    * Payoffs are ``(host, player)`` per agent class:

      +----------+----------+------------------+
      | Proposal | Outcome  | (host, player)   |
      +==========+==========+==================+
      | A        | accept   | (n, -1)          |
      | A        | reject   | (-n, 1)          |
      | B        | accept   | (1, 1)           |
      | B        | reject   | (-1, -1)         |
      +----------+----------+------------------+

    where ``n = num_players + 1`` (players plus host).

    Round API: ``step_propose`` → ``step_vote``, or ``step(action)`` keyed by phase.
  """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ProposalVoteConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or ProposalVoteConfig()
        self.rng = np.random.default_rng(seed)
        self._build_spaces()
        self._init_state()

    def _build_spaces(self) -> None:
        c = self.cfg
        n = c.num_players

        self.host_propose_action_space = spaces.Discrete(2)
        self.player_vote_action_space = spaces.MultiDiscrete([2] * n)

        self.player_observation_space = spaces.Dict({
            "current": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(n, c.current_player_dim),
                dtype=np.float32,
            ),
            "history": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(n, c.history_window, c.hist_player_dim),
                dtype=np.float32,
            ),
        })
        self.host_observation_space = spaces.Dict({
            "current": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(c.current_host_dim,),
                dtype=np.float32,
            ),
            "players": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n, c.host_player_state_dim),
                dtype=np.float32,
            ),
            "history": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(c.history_window, c.hist_host_dim),
                dtype=np.float32,
            ),
            "vote_history": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n, c.history_window),
                dtype=np.float32,
            ),
        })
        self.observation_space = spaces.Dict({
            "players": self.player_observation_space,
            "host": self.host_observation_space,
        })
        self.action_space = spaces.Dict({
            "host_proposal": self.host_propose_action_space,
            "player_votes": self.player_vote_action_space,
        })

    def _init_state(self) -> None:
        c = self.cfg
        n = c.num_players
        self.round_idx = 0
        self.phase = Phase.PROPOSE
        self.host_cumulative_reward = 0.0
        self.player_cumulative_rewards = np.zeros(n, dtype=np.float32)

        self.current_proposal: int = -1
        self.current_votes: np.ndarray = np.full(n, -1, dtype=np.int32)

        w = c.history_window
        self.hist_own_vote = np.full((w, n), -1.0, dtype=np.float32)
        self.hist_passed = np.full((w,), -1.0, dtype=np.float32)
        self.hist_player_rewards = np.zeros((w, n), dtype=np.float32)
        self.hist_host_reward = np.zeros((w,), dtype=np.float32)
        self.hist_frac_accept = np.full((w,), -1.0, dtype=np.float32)
        self.hist_proposal = np.full((w,), -1.0, dtype=np.float32)
        self.hist_votes = np.full((w, n), -1.0, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        return self._get_observations(), self._get_info()

    @staticmethod
    def _majority_accept(votes: np.ndarray) -> bool:
        """True if accept wins or ties (tie → pass)."""
        accept = int(np.sum(votes == Vote.ACCEPT))
        reject = int(np.sum(votes == Vote.REJECT))
        return accept >= reject

    @staticmethod
    def _settle(
        proposal: int,
        accepted: bool,
        n_group: int,
    ) -> tuple[float, float]:
        """Return (host_reward, player_reward) for this round."""
        if proposal == Proposal.A:
            if accepted:
                return float(n_group), -1.0
            return float(-n_group), 1.0
        if proposal == Proposal.B:
            if accepted:
                return 1.0, 1.0
            return -1.0, -1.0
        raise ValueError(f"invalid proposal {proposal}")

    def step_propose(self, proposal: int) -> tuple[dict, dict, bool, bool, dict]:
        if self.phase != Phase.PROPOSE:
            raise RuntimeError(
                f"step_propose() requires Phase.PROPOSE; got {self.phase.name}"
            )
        proposal = int(np.clip(proposal, 0, 1))
        self.current_proposal = proposal
        self.current_votes = np.full(self.cfg.num_players, -1, dtype=np.int32)
        self.phase = Phase.VOTE
        return self._get_observations(), {}, False, False, self._get_info()

    def step_vote(
        self,
        player_votes: np.ndarray,
    ) -> tuple[dict, dict[str, Any], bool, bool, dict]:
        if self.phase != Phase.VOTE:
            raise RuntimeError(
                f"step_vote() requires Phase.VOTE; got {self.phase.name}"
            )
        c = self.cfg
        player_votes = np.asarray(player_votes, dtype=np.int32)
        if player_votes.shape != (c.num_players,):
            raise ValueError(
                f"player_votes must have shape ({c.num_players},), got {player_votes.shape}"
            )

        votes = np.clip(player_votes, Vote.REJECT, Vote.ACCEPT).astype(np.int32)
        self.current_votes = votes
        accepted = self._majority_accept(votes)
        host_r, player_r = self._settle(
            self.current_proposal, accepted, c.n
        )

        player_rewards = np.full(c.num_players, player_r, dtype=np.float32)
        self.host_cumulative_reward += host_r
        self.player_cumulative_rewards += player_rewards

        frac_accept = float(np.mean(votes == Vote.ACCEPT))
        self._push_history(
            proposal=float(self.current_proposal),
            votes=votes.astype(np.float32),
            passed=float(accepted),
            host_reward=host_r,
            player_rewards=player_rewards,
            frac_accept=frac_accept,
        )

        self.round_idx += 1
        terminated = self.round_idx >= c.max_rounds
        truncated = False

        self.current_proposal = -1
        self.current_votes = np.full(c.num_players, -1, dtype=np.int32)
        if not terminated:
            self.phase = Phase.PROPOSE

        rewards: dict[str, Any] = {
            "players": player_rewards,
            "host": float(host_r),
        }
        info = self._get_info()
        info["proposal"] = int(self.hist_proposal[-1])
        info["proposal_name"] = "A" if info["proposal"] == 0 else "B"
        info["votes"] = votes.copy()
        info["accepted"] = accepted
        info["host_reward"] = host_r
        info["player_reward_each"] = player_r
        return self._get_observations(), rewards, terminated, truncated, info

    def step(self, action: dict) -> tuple:
        if self.phase == Phase.PROPOSE:
            if "host_proposal" not in action and "proposal" not in action:
                raise KeyError("PROPOSE phase requires 'host_proposal' or 'proposal'")
            prop = action.get("host_proposal", action.get("proposal"))
            return self.step_propose(prop)
        if "player_votes" not in action and "votes" not in action:
            raise KeyError("VOTE phase requires 'player_votes' or 'votes'")
        votes = action.get("player_votes", action.get("votes"))
        return self.step_vote(votes)

    def _get_player_obs(self) -> dict[str, np.ndarray]:
        c = self.cfg
        n = c.num_players
        current = np.zeros((n, c.current_player_dim), dtype=np.float32)
        in_vote = float(self.phase == Phase.VOTE)
        last_passed = self.hist_passed[-1]
        for i in range(n):
            last_vote = self.hist_own_vote[-1, i]
            last_reward = self.hist_player_rewards[-1, i]
            current[i] = [in_vote, last_passed, last_vote, last_reward]

        history = np.stack([
            self.hist_own_vote,
            np.broadcast_to(self.hist_passed[:, None], (c.history_window, n)),
            self.hist_player_rewards,
            np.broadcast_to(self.hist_frac_accept[:, None], (c.history_window, n)),
            np.broadcast_to(self.hist_host_reward[:, None], (c.history_window, n)),
        ], axis=2).transpose(1, 0, 2).astype(np.float32)

        return {"current": current, "history": history}

    def _get_host_obs(self) -> dict[str, np.ndarray]:
        c = self.cfg
        prop_a = 1.0 if self.current_proposal == Proposal.A else 0.0
        prop_b = 1.0 if self.current_proposal == Proposal.B else 0.0
        if self.current_proposal < 0:
            prop_a = prop_b = 0.0
        round_frac = (
            float(self.round_idx / max(c.max_rounds, 1))
            if c.max_rounds > 0
            else 0.0
        )
        current = np.array(
            [self.host_cumulative_reward, prop_a, prop_b, round_frac],
            dtype=np.float32,
        )

        last_votes = np.where(
            self.current_votes >= 0,
            self.current_votes.astype(np.float32),
            self.hist_votes[-1],
        )
        players = last_votes[:, None].astype(np.float32)

        history = np.stack([
            self.hist_proposal,
            self.hist_passed,
            self.hist_host_reward,
            self.hist_frac_accept,
        ], axis=1).astype(np.float32)

        vote_history = self.hist_votes.T.astype(np.float32)

        return {
            "current": current,
            "players": players,
            "history": history,
            "vote_history": vote_history,
        }

    def _get_observations(self) -> dict[str, dict]:
        return {
            "players": self._get_player_obs(),
            "host": self._get_host_obs(),
        }

    def _get_info(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "phase": int(self.phase),
            "phase_name": self.phase.name,
            "round": self.round_idx,
            "n_group": c.n,
            "host_cumulative_reward": float(self.host_cumulative_reward),
            "player_cumulative_rewards": self.player_cumulative_rewards.copy(),
            "current_proposal": self.current_proposal,
        }

    def _push_history(
        self,
        *,
        proposal: float,
        votes: np.ndarray,
        passed: float,
        host_reward: float,
        player_rewards: np.ndarray,
        frac_accept: float,
    ) -> None:
        self.hist_proposal = np.roll(self.hist_proposal, -1)
        self.hist_passed = np.roll(self.hist_passed, -1)
        self.hist_host_reward = np.roll(self.hist_host_reward, -1)
        self.hist_frac_accept = np.roll(self.hist_frac_accept, -1)
        self.hist_own_vote = np.roll(self.hist_own_vote, -1, axis=0)
        self.hist_player_rewards = np.roll(self.hist_player_rewards, -1, axis=0)
        self.hist_votes = np.roll(self.hist_votes, -1, axis=0)

        self.hist_proposal[-1] = proposal
        self.hist_passed[-1] = passed
        self.hist_host_reward[-1] = host_reward
        self.hist_frac_accept[-1] = frac_accept
        self.hist_own_vote[-1] = votes
        self.hist_player_rewards[-1] = player_rewards
        self.hist_votes[-1] = votes
