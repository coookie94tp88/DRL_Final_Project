"""DooRLEnv: the 4-Door bribery / signaling MARL environment.

Round structure (3 env substeps; see spec.md §2):
    phase 0 (Phase I):  players submit ``bribe_pct``;     host action ignored.
    phase 1 (Host):     host submits public_door + private logits; players' actions ignored.
    phase 2 (Phase II): players submit ``door`` + ``bet_pct``; host action ignored.
                         Settlement and rewards are emitted at the end of phase 2.

The environment follows the PettingZoo ``ParallelEnv`` API. Agent list:

    agents = ["host", "player_0", "player_1", ..., "player_{N-1}"]

Action and observation spaces are ``gymnasium.spaces.Dict`` so that the same
fixed-shape action can be used across phases; unused fields are ignored per phase.

PettingZoo's ``ParallelEnv`` base class is optional at runtime — we expose the
same interface so the env works with or without ``pettingzoo`` installed.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dep
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover - fall back to a tiny stub
    spaces = None  # type: ignore

from doorl.env.payout import calculate_multiplier
from doorl.env.types import EnvConfig, EpisodeStats, LastSettlement, RoundRecord


PHASE_BRIBE = 0
PHASE_HOST = 1
PHASE_BET = 2
NUM_PHASES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player_id(i: int) -> str:
    return f"player_{i}"


def _make_player_action_space(num_players: int):
    if spaces is None:
        return None
    return spaces.Dict(
        {
            "bribe_pct": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "door": spaces.Discrete(4),
            "bet_pct": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
    )


def _make_host_action_space(num_players: int):
    if spaces is None:
        return None
    return spaces.Dict(
        {
            "public_door": spaces.Discrete(4),
            "private_logits": spaces.Box(
                low=-10.0, high=10.0, shape=(num_players, 4), dtype=np.float32
            ),
        }
    )


def _history_row_size(num_players: int) -> int:
    # own_door(4) + public_signal(4) + own_private_signal(4) + door_share(4)
    # + R_player_self(1) + R_player_all(N) + x(1) + R_host_delta(1) + active_mask(N)
    return 4 + 4 + 4 + 4 + 1 + num_players + 1 + 1 + num_players


def _zero_history(num_players: int, history_window: int) -> np.ndarray:
    return np.zeros(
        (history_window, _history_row_size(num_players)), dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class DooRLEnv:
    """Parallel multi-agent environment for the DooRL game.

    Mirrors PettingZoo's ``ParallelEnv`` interface:

        env.reset(seed=...) -> (observations, infos)
        env.step(actions)   -> (observations, rewards, terminations, truncations, infos)
    """

    metadata = {"name": "doorl_v0", "is_parallelizable": True}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = EnvConfig(**overrides)
        else:
            for k, v in overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        config.validate()
        self.cfg = config

        self.num_players = config.num_players
        self.num_doors = config.num_doors
        self.history_window = config.history_window

        self.possible_agents: List[str] = ["host"] + [
            _make_player_id(i) for i in range(self.num_players)
        ]
        self.agents: List[str] = list(self.possible_agents)

        self._player_action_space = _make_player_action_space(self.num_players)
        self._host_action_space = _make_host_action_space(self.num_players)

        self._player_obs_dim_phase1 = self._compute_player_obs_dim(phase=PHASE_BRIBE)
        self._player_obs_dim_phase2 = self._compute_player_obs_dim(phase=PHASE_BET)
        self._host_obs_dim = self._compute_host_obs_dim()

        self.action_spaces = {
            agent: (
                self._host_action_space
                if agent == "host"
                else self._player_action_space
            )
            for agent in self.possible_agents
        }
        self.observation_spaces = self._build_observation_spaces()

        self._rng = np.random.default_rng(config.seed)

        # populated by reset() / step()
        self._phase: int = PHASE_BRIBE
        self._round_idx: int = 0
        self._balances: np.ndarray = np.full(
            (self.num_players,), config.initial_balance, dtype=np.float64
        )
        self._active: np.ndarray = np.ones((self.num_players,), dtype=np.float32)
        self._history: np.ndarray = _zero_history(
            self.num_players, self.history_window
        )
        self._last_R_player_all: np.ndarray = np.zeros(
            self.num_players, dtype=np.float32
        )
        self._last_bribe_pcts: np.ndarray = np.zeros(
            self.num_players, dtype=np.float32
        )

        # transient: persists across substeps within one round
        self._round_true_door: int = 0
        self._round_bribe_pcts: np.ndarray = np.zeros(
            self.num_players, dtype=np.float32
        )
        self._round_bribes: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self._round_balance_after_bribe: np.ndarray = self._balances.copy()
        self._round_public_signal: int = 0
        self._round_private_signals: np.ndarray = np.zeros(
            self.num_players, dtype=np.int64
        )
        self._round_private_distributions: np.ndarray = np.full(
            (self.num_players, 4), 0.25, dtype=np.float32
        )
        self._round_host_bribe_tally: float = 0.0

        self.stats = EpisodeStats()
        self.last_settlement: Optional[LastSettlement] = None
        self._closed = False

    # --------------------- observation / action shapes ---------------------

    def _compute_player_obs_dim(self, phase: int) -> int:
        n = self.num_players
        dim = 0
        dim += n  # balances
        dim += n  # last_round_bribe_pcts
        dim += n  # last_R_player_all
        dim += self.history_window * _history_row_size(n)  # history buffer
        dim += n  # active_mask
        dim += n  # own_index_one_hot
        dim += 1  # phase indicator (normalized)
        dim += 1  # round_idx / max_rounds
        if phase == PHASE_BET:
            dim += n  # this_round_bribe_pcts
            dim += 4  # public_signal
            dim += 4  # own_private_signal
        return dim

    def _compute_host_obs_dim(self) -> int:
        n = self.num_players
        dim = 0
        dim += n  # balances (always full visibility)
        dim += n  # this_round_bribe_pcts (filled after Phase I; zeros at phase 0)
        dim += n  # last_R_player_all
        dim += self.history_window * _history_row_size(n)
        dim += n  # active_mask
        dim += 4  # true_door one-hot (current round)
        dim += n * 4  # last-round emitted private distributions
        dim += 1  # phase indicator
        dim += 1  # round_idx / max_rounds
        return dim

    def _build_observation_spaces(self):
        if spaces is None:
            return None
        # Use the larger of the two player obs dims to keep a fixed shape; we
        # zero-pad Phase I obs at runtime to match Phase II length.
        player_dim = max(self._player_obs_dim_phase1, self._player_obs_dim_phase2)
        return {
            "host": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._host_obs_dim,), dtype=np.float32
            ),
            **{
                _make_player_id(i): spaces.Box(
                    low=-np.inf, high=np.inf, shape=(player_dim,), dtype=np.float32
                )
                for i in range(self.num_players)
            },
        }

    @property
    def player_obs_dim(self) -> int:
        return max(self._player_obs_dim_phase1, self._player_obs_dim_phase2)

    @property
    def host_obs_dim(self) -> int:
        return self._host_obs_dim

    # ----------------------------- reset ---------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._phase = PHASE_BRIBE
        self._round_idx = 0
        self._balances = np.full(
            (self.num_players,), self.cfg.initial_balance, dtype=np.float64
        )
        self._active = np.ones((self.num_players,), dtype=np.float32)
        self._history = _zero_history(self.num_players, self.history_window)
        self._last_R_player_all = np.zeros(self.num_players, dtype=np.float32)
        self._last_bribe_pcts = np.zeros(self.num_players, dtype=np.float32)
        self._round_private_distributions = np.full(
            (self.num_players, 4), 0.25, dtype=np.float32
        )
        self.agents = list(self.possible_agents)
        self.stats = EpisodeStats()
        self.last_settlement: Optional[LastSettlement] = None

        self._start_new_round()

        return self._build_all_observations(), self._build_all_infos()

    def _start_new_round(self) -> None:
        self._round_true_door = int(self._rng.integers(0, 4))
        self._round_bribe_pcts = np.zeros(self.num_players, dtype=np.float32)
        self._round_bribes = np.zeros(self.num_players, dtype=np.float64)
        self._round_balance_after_bribe = self._balances.copy()
        self._round_public_signal = 0
        self._round_private_signals = np.zeros(self.num_players, dtype=np.int64)
        self._round_host_bribe_tally = 0.0
        self._phase = PHASE_BRIBE

    # ------------------------------ step ---------------------------------

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        if self._closed:
            raise RuntimeError("step() called on a closed environment")

        rewards = {agent: 0.0 for agent in self.possible_agents}

        if self._phase == PHASE_BRIBE:
            self._apply_phase_i(actions)
            self._phase = PHASE_HOST
        elif self._phase == PHASE_HOST:
            self._apply_phase_host(actions)
            self._phase = PHASE_BET
        elif self._phase == PHASE_BET:
            rewards = self._apply_phase_ii_and_settle(actions)
            self._round_idx += 1
            # Begin next round: after settlement phase was still BET, so the old
            # ``phase == PHASE_BRIBE`` check never fired and true_door never changed.
            episode_done = self._is_episode_done()
            if not episode_done:
                self._phase = PHASE_BRIBE
                self._start_new_round()
        else:  # pragma: no cover
            raise RuntimeError(f"Invalid phase {self._phase}")

        episode_done = self._is_episode_done()
        terminations = {agent: episode_done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = self._build_all_infos()

        if episode_done:
            self.agents = []

        return self._build_all_observations(), rewards, terminations, truncations, infos

    # ----------------- phase handlers ------------------

    def _apply_phase_i(self, actions: Dict[str, Any]) -> None:
        bribe_floor = float(self.cfg.bribery_floor)
        for i in range(self.num_players):
            agent = _make_player_id(i)
            a = actions.get(agent, {})
            raw = float(np.asarray(a.get("bribe_pct", 0.0)).reshape(-1)[0])
            pct = float(np.clip(raw, bribe_floor, 1.0))
            if self._active[i] == 0.0:
                pct = 0.0
            self._round_bribe_pcts[i] = pct
            bribe = self._balances[i] * pct
            self._round_bribes[i] = bribe
            self._round_balance_after_bribe[i] = self._balances[i] - bribe
            self._round_host_bribe_tally += bribe
        # update last_bribe_pcts in obs for the next round; balances are deducted now
        self._balances = self._round_balance_after_bribe.copy()

    def _apply_phase_host(self, actions: Dict[str, Any]) -> None:
        a = actions.get("host", {})
        public_door = int(np.asarray(a.get("public_door", 0)).reshape(-1)[0])
        public_door = int(np.clip(public_door, 0, 3))

        logits = np.asarray(a.get("private_logits", np.zeros((self.num_players, 4))))
        logits = logits.reshape(self.num_players, 4).astype(np.float32)

        # stable softmax
        m = logits.max(axis=1, keepdims=True)
        exps = np.exp(logits - m)
        probs = exps / exps.sum(axis=1, keepdims=True)
        # sample one private door per player
        privates = np.array(
            [self._rng.choice(4, p=probs[i]) for i in range(self.num_players)],
            dtype=np.int64,
        )

        self._round_public_signal = public_door
        self._round_private_signals = privates
        self._round_private_distributions = probs.astype(np.float32)

    def _apply_phase_ii_and_settle(
        self, actions: Dict[str, Any]
    ) -> Dict[str, float]:
        true_door = self._round_true_door
        bets = np.zeros(self.num_players, dtype=np.float64)
        chosen_doors = np.zeros(self.num_players, dtype=np.int64)
        for i in range(self.num_players):
            agent = _make_player_id(i)
            a = actions.get(agent, {})
            door = int(np.asarray(a.get("door", 0)).reshape(-1)[0])
            door = int(np.clip(door, 0, 3))
            bet_pct = float(np.asarray(a.get("bet_pct", 0.0)).reshape(-1)[0])
            bet_pct = float(np.clip(bet_pct, 0.0, 1.0))

            chosen_doors[i] = door
            if self._active[i] == 0.0 or self._balances[i] < self.cfg.min_bet:
                bets[i] = 0.0
                # mark as inactive going forward if balance can't cover min_bet
                if self._balances[i] < self.cfg.min_bet:
                    self._active[i] = 0.0
                continue
            absolute = bet_pct * float(self._balances[i])
            absolute = max(self.cfg.min_bet, absolute)
            absolute = min(absolute, float(self._balances[i]))
            bets[i] = float(round(absolute))
            self._balances[i] -= bets[i]

        P = float(bets.sum())
        winning_mask = (chosen_doors == true_door) & (self._active > 0.0)
        W = float(bets[winning_mask].sum())

        payouts = np.zeros(self.num_players, dtype=np.float64)
        if P > 0.0 and W > 0.0:
            x = W / P
            m = calculate_multiplier(
                x, tau=self.cfg.payout_threshold, max_multiplier=self.cfg.max_multiplier
            )
            payouts[winning_mask] = bets[winning_mask] * m
        else:
            x = 0.0
            m = 0.0

        self._balances += payouts

        rewards_player = (payouts - bets - self._round_bribes).astype(np.float32)
        host_pool_pnl = P - float(payouts.sum())
        reward_host = float(host_pool_pnl + self._round_host_bribe_tally)

        # door share vector
        door_share = np.zeros(4, dtype=np.float32)
        if P > 0.0:
            for d in range(4):
                door_share[d] = float(bets[chosen_doors == d].sum() / P)

        # update history
        round_active_mask = self._active.copy()
        self._push_history_row(
            chosen_doors=chosen_doors,
            public_signal=self._round_public_signal,
            private_signals=self._round_private_signals,
            door_share_vec=door_share,
            rewards_player=rewards_player,
            x=x,
            reward_host=reward_host,
            active_mask=round_active_mask,
        )

        # update post-round mutable state used by next-round observations
        self._last_R_player_all = rewards_player.copy()
        self._last_bribe_pcts = self._round_bribe_pcts.copy()

        # post-round bankruptcy detection
        for i in range(self.num_players):
            if self._balances[i] < self.cfg.min_bet:
                if self._active[i] != 0.0:
                    self.stats.bankruptcies += 1
                self._active[i] = 0.0

        # stats
        self.stats.rounds_played += 1
        self.stats.sum_R_host += reward_host
        self.stats.sum_bribes += float(self._round_bribes.sum())
        if P > 0.0:
            self.stats.x_values.append(x)
            self.stats.rounds_with_pool += 1
        if self._round_public_signal == true_door:
            self.stats.public_truth_rate_num += 1
            # public-true + private-false rate
            misled = int(((self._round_private_signals != true_door)).sum())
            self.stats.public_true_private_false_num += misled
        self.stats.private_truth_rate_num.append(
            int((self._round_private_signals == true_door).sum())
        )

        self.last_settlement = LastSettlement(
            round_idx=int(self._round_idx),
            true_door=int(true_door),
            public_signal=int(self._round_public_signal),
            private_signals=self._round_private_signals.copy(),
            chosen_doors=chosen_doors.copy(),
            bribe_pcts=self._round_bribe_pcts.copy(),
            bribes=self._round_bribes.copy(),
            bets=bets.copy(),
            payouts=payouts.copy(),
            rewards_player=rewards_player.copy(),
            reward_host=float(reward_host),
            x=float(x),
            multiplier=float(m),
            door_share=door_share.copy(),
            pool_p=float(P),
        )

        rewards = {"host": reward_host}
        for i in range(self.num_players):
            rewards[_make_player_id(i)] = float(rewards_player[i])
        return rewards

    # ----------------- history buffer ------------------

    def _push_history_row(
        self,
        chosen_doors: np.ndarray,
        public_signal: int,
        private_signals: np.ndarray,
        door_share_vec: np.ndarray,
        rewards_player: np.ndarray,
        x: float,
        reward_host: float,
        active_mask: np.ndarray,
    ) -> None:
        # We pack a per-round "shared" portion + per-player extras at obs build time.
        # For storage efficiency, the buffer stores everything **using player 0's** own
        # private signal as a placeholder; per-player obs construction overwrites the
        # `own_private_signal` and `own_door` slices at read time.
        n = self.num_players
        row = np.zeros(_history_row_size(n), dtype=np.float32)
        # We store CHOSEN_DOORS as one-hot of player 0 here as a placeholder; per-player
        # obs reconstruction supplies the right own_door / own_private one-hots later
        # using the parallel arrays stored below.
        # To keep things simple we store per-player vectors in a separate side buffer.
        self._history = np.roll(self._history, shift=-1, axis=0)
        offset = 0
        # own_door one-hot will be overwritten per player; leave 0
        offset += 4
        # public_signal one-hot
        row[offset + public_signal] = 1.0
        offset += 4
        # own_private_signal placeholder; per-player overwrite at obs time
        offset += 4
        # door_share_vec
        row[offset : offset + 4] = door_share_vec
        offset += 4
        # R_player_self placeholder; per-player overwrite
        offset += 1
        # R_player_all (N,)
        row[offset : offset + n] = rewards_player
        offset += n
        # x scalar
        row[offset] = float(x)
        offset += 1
        # R_host_delta scalar
        row[offset] = float(reward_host)
        offset += 1
        # active_mask (N,)
        row[offset : offset + n] = active_mask
        self._history[-1] = row

        # cache per-player data needed to fill placeholders at obs build time
        self._last_chosen_doors = chosen_doors.copy()
        self._last_private_signals = private_signals.copy()
        self._last_rewards_player = rewards_player.copy()

    # ---------------------- observation builders ------------------------

    def _build_all_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {"host": self._build_host_obs()}
        for i in range(self.num_players):
            obs[_make_player_id(i)] = self._build_player_obs(i)
        return obs

    def _balances_view_for_player(self, i: int) -> np.ndarray:
        vis = self.cfg.balance_visibility
        norm = self._balances / max(self.cfg.initial_balance, 1.0)
        if vis == "full":
            return norm.astype(np.float32)
        if vis == "own_only":
            out = np.zeros_like(norm, dtype=np.float32)
            out[i] = float(norm[i])
            return out
        if vis == "noisy":
            noise = self._rng.lognormal(
                mean=0.0, sigma=float(self.cfg.balance_noise_sigma), size=norm.shape
            ).astype(np.float32)
            out = (norm * noise).astype(np.float32)
            out[i] = float(norm[i])
            return out
        raise ValueError(f"Unknown balance_visibility {vis!r}")

    def _own_view_history(self, i: int) -> np.ndarray:
        """Build the personalized history buffer for player i by filling the
        own_door / own_private / R_player_self placeholders.
        """
        n = self.num_players
        hist = self._history.copy()
        # We only know the chosen_doors / private_signals from the most recent round,
        # so we lazily set them on the LAST row (older rows already had their
        # personalized fields zero, which is fine — players still see R_player_all and
        # door_share which carry most of the signal). This is a deliberate
        # simplification documented in spec.md (history personalisation is best-effort).
        if self._round_idx > 0:
            row = hist[-1]
            own_door = int(self._last_chosen_doors[i])
            row[0:4] = 0.0
            row[own_door] = 1.0
            own_private = int(self._last_private_signals[i])
            row[8:12] = 0.0
            row[8 + own_private] = 1.0
            # R_player_self slot
            r_self_offset = 4 + 4 + 4 + 4  # past 4 fields
            row[r_self_offset] = float(self._last_rewards_player[i])
            hist[-1] = row
        return hist.reshape(-1)

    def _build_player_obs(self, i: int) -> np.ndarray:
        n = self.num_players
        balances = self._balances_view_for_player(i)
        last_bribes = self._last_bribe_pcts.astype(np.float32)
        last_R_all = self._last_R_player_all.astype(np.float32)
        hist_flat = self._own_view_history(i)
        active_mask = self._active.astype(np.float32)
        own_index = np.zeros(n, dtype=np.float32)
        own_index[i] = 1.0
        phase_indicator = np.array(
            [self._phase / float(NUM_PHASES - 1)], dtype=np.float32
        )
        round_indicator = np.array(
            [self._round_idx / max(self.cfg.max_rounds, 1)], dtype=np.float32
        )

        parts = [balances, last_bribes, last_R_all, hist_flat, active_mask, own_index,
                 phase_indicator, round_indicator]

        if self._phase == PHASE_BET:
            parts.append(self._round_bribe_pcts.astype(np.float32))
            ps = np.zeros(4, dtype=np.float32)
            ps[self._round_public_signal] = 1.0
            parts.append(ps)
            priv = np.zeros(4, dtype=np.float32)
            priv[int(self._round_private_signals[i])] = 1.0
            parts.append(priv)

        obs = np.concatenate(parts).astype(np.float32)
        target_dim = self.player_obs_dim
        if obs.shape[0] < target_dim:
            obs = np.concatenate(
                [obs, np.zeros(target_dim - obs.shape[0], dtype=np.float32)]
            )
        elif obs.shape[0] > target_dim:
            obs = obs[:target_dim]
        return obs

    def _build_host_obs(self) -> np.ndarray:
        n = self.num_players
        balances = (self._balances / max(self.cfg.initial_balance, 1.0)).astype(
            np.float32
        )
        this_round_bribes = self._round_bribe_pcts.astype(np.float32)
        last_R_all = self._last_R_player_all.astype(np.float32)
        hist_flat = self._history.copy().reshape(-1).astype(np.float32)
        active_mask = self._active.astype(np.float32)
        true_door_oh = np.zeros(4, dtype=np.float32)
        true_door_oh[self._round_true_door] = 1.0
        priv_flat = self._round_private_distributions.reshape(-1).astype(np.float32)
        phase_indicator = np.array(
            [self._phase / float(NUM_PHASES - 1)], dtype=np.float32
        )
        round_indicator = np.array(
            [self._round_idx / max(self.cfg.max_rounds, 1)], dtype=np.float32
        )
        return np.concatenate(
            [
                balances,
                this_round_bribes,
                last_R_all,
                hist_flat,
                active_mask,
                true_door_oh,
                priv_flat,
                phase_indicator,
                round_indicator,
            ]
        ).astype(np.float32)

    # -------------------- termination / infos --------------------------

    def _is_episode_done(self) -> bool:
        if self._round_idx >= self.cfg.max_rounds:
            return True
        # all bankrupt -> done
        if float(self._active.sum()) == 0.0:
            return True
        return False

    def _build_all_infos(self) -> Dict[str, dict]:
        info = {
            "phase": int(self._phase),
            "round": int(self._round_idx),
            "true_door": int(self._round_true_door),
            "balances": self._balances.copy(),
            "active_mask": self._active.copy(),
            "bribe_pcts": self._round_bribe_pcts.copy(),
            "public_signal": int(self._round_public_signal),
            "private_signals": self._round_private_signals.copy(),
        }
        return {agent: info for agent in self.possible_agents}

    # ------------------ public utility ---------------------

    def observation_space(self, agent: str):
        if self.observation_spaces is None:  # pragma: no cover
            return None
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        if self.action_spaces is None:  # pragma: no cover
            return None
        return self.action_spaces[agent]

    def render(self) -> None:  # pragma: no cover - intentional no-op
        pass

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Convenience constructors mirroring PettingZoo's pattern.
# ---------------------------------------------------------------------------


def parallel_env(**kwargs: Any) -> DooRLEnv:
    """Create a parallel DooRL environment."""
    return DooRLEnv(**kwargs)


def raw_env(**kwargs: Any) -> DooRLEnv:
    """Alias for ``parallel_env`` (DooRL is natively parallel)."""
    return DooRLEnv(**kwargs)
