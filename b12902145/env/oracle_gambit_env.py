"""
OracleGambit: Strategic Information Manipulation in Multi-Agent RL
==================================================================
A PettingZoo-style (AEC) MARL environment implementing the 4-Door Game
with Bribery, Signaling, and Dynamic Payout mechanics.

Turn order per round:
  Phase I  -> all players submit bribe simultaneously
            -> host observes bribes and emits signals
  Phase II -> all players submit (door, bet) simultaneously
            -> environment settles and distributes rewards
"""

from __future__ import annotations

import collections
import functools
from typing import Any

import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Optional PettingZoo base-class import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import wrappers
    from pettingzoo.utils.agent_selector import agent_selector
    _HAS_PETTINGZOO = True
except ImportError:  # pragma: no cover
    _HAS_PETTINGZOO = False

    class AECEnv:  # type: ignore[no-redef]
        """Minimal stub so the file is importable without PettingZoo."""

        metadata: dict = {}

        def __init__(self) -> None:
            pass


# ---------------------------------------------------------------------------
# Global constants (can be overridden via __init__ kwargs)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "num_players": 10,
    "num_doors": 4,
    "initial_balance": 1000.0,
    # Break-even threshold: if winning ratio >= PAYOUT_THRESHOLD, host loses
    "payout_threshold": 0.25,
    "history_window": 50,
    # Welfare payment given to bankrupt players so they can keep acting
    "welfare_amount": 10.0,
    # Host's rake on the total pool (fee paid regardless of outcome)
    "fee_rate": 0.05,
    # Maximum rounds per episode (0 = unlimited)
    "max_rounds": 500,
    # Random seed
    "seed": None,
}

# Padding token used when the history buffer is not yet full
PAD_VALUE: float = -1.0

# ---------------------------------------------------------------------------
# History feature sizes (used to build observation vectors)
# ---------------------------------------------------------------------------
# Per-round player history features:
#   door_choice (1), private_signal (1), public_signal (1),
#   bribe (1), bet (1), balance (1), won (1), payout (1),
#   door_ratios (NUM_DOORS)
#   = 8 + NUM_DOORS
_PLAYER_HIST_BASE = 8  # non-door-ratio features


def _player_hist_size(num_doors: int) -> int:
    return _PLAYER_HIST_BASE + num_doors


# Per-round host history features:
#   correct_door (1), public_signal (1),
#   per-player: bribe (1), private_signal (1), balance (1), door_choice (1), won (1)
#   = 2 + 5 * NUM_PLAYERS
_HOST_HIST_BASE = 2
_HOST_HIST_PER_PLAYER = 5


def _host_hist_size(num_players: int) -> int:
    return _HOST_HIST_BASE + _HOST_HIST_PER_PLAYER * num_players


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pad_history(
    deque_buf: collections.deque,
    window: int,
    feat_size: int,
) -> np.ndarray:
    """Return (window, feat_size) float32 array; pads with PAD_VALUE."""
    buf = np.full((window, feat_size), PAD_VALUE, dtype=np.float32)
    data = list(deque_buf)  # oldest first
    n = len(data)
    if n > 0:
        buf[-n:] = np.array(data, dtype=np.float32)
    return buf


def _build_attention_mask(deque_buf: collections.deque, window: int) -> np.ndarray:
    """Return bool mask of shape (window,); True = valid, False = padding."""
    mask = np.zeros(window, dtype=bool)
    n = len(deque_buf)
    if n > 0:
        mask[-n:] = True
    return mask


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class OracleGambitEnv(AECEnv):
    """
    OracleGambit AEC environment.

    Agent IDs
    ---------
    * ``"host"``          – the Oracle
    * ``"player_0"`` …    – the N players

    Action spaces
    -------------
    * host (Phase I only):
        Box([public_signal, private_signal_0, …, private_signal_{N-1}])
        All values in [0, num_doors-1] (continuous; rounded internally).

    * player Phase I (bribe):
        Box([bribe_fraction])  in [0, 1] as a fraction of current balance.

    * player Phase II (door + bet):
        Box([door_fraction, bet_fraction])
        door_fraction in [0, 1] → mapped to 0…num_doors-1
        bet_fraction  in [0, 1] → fraction of remaining balance

    Observation spaces
    ------------------
    Each agent receives a dict with keys:
        "history"      – float32 (window, feat_size)
        "attn_mask"    – bool    (window,)
        "current"      – float32 (current_feat_size,)

    Rewards
    -------
    Delivered at the END of Phase II (i.e., after settlement).
    """

    metadata = {"render_modes": ["human"], "name": "oracle_gambit_v0"}

    # ------------------------------------------------------------------
    def __init__(self, **cfg_override) -> None:
        super().__init__()

        # Merge defaults with overrides
        cfg = {**DEFAULT_CONFIG, **cfg_override}
        self.num_players: int = int(cfg["num_players"])
        self.num_doors: int = int(cfg["num_doors"])
        self.initial_balance: float = float(cfg["initial_balance"])
        self.payout_threshold: float = float(cfg["payout_threshold"])
        self.history_window: int = int(cfg["history_window"])
        self.welfare_amount: float = float(cfg["welfare_amount"])
        self.fee_rate: float = float(cfg["fee_rate"])
        self.max_rounds: int = int(cfg["max_rounds"])
        self._seed = cfg["seed"]

        self._rng = np.random.default_rng(self._seed)

        # Agent identifiers
        self.possible_agents: list[str] = ["host"] + [
            f"player_{i}" for i in range(self.num_players)
        ]

        # Precompute feature sizes
        self._player_hist_feat = _player_hist_size(self.num_doors)
        self._host_hist_feat = _host_hist_size(self.num_players)

        # Build spaces once (they never change)
        self._build_spaces()

        # Internal state (populated in reset)
        self.agents: list[str] = []
        self._phase: int = 0          # 0 = Phase I, 1 = Phase II
        self._round: int = 0
        self._correct_door: int = 0

        # Balances and accumulated values for current round
        self._balances: dict[str, float] = {}
        self._bribes: dict[str, float] = {}
        self._bets: dict[str, float] = {}
        self._door_choices: dict[str, int] = {}
        self._public_signal: int = 0
        self._private_signals: dict[str, int] = {}

        # History buffers
        self._player_hist: dict[str, collections.deque] = {}
        self._host_hist: collections.deque = collections.deque(
            maxlen=self.history_window
        )
        self._host_cumulative_profit: float = 0.0

        # PettingZoo required dicts
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self._cumulative_rewards: dict[str, float] = {}

        if _HAS_PETTINGZOO:
            self._agent_selector = agent_selector(self.possible_agents)

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------

    def _build_spaces(self) -> None:
        ph = self._player_hist_feat
        hh = self._host_hist_feat
        L = self.history_window

        # ------ HOST ------
        # observation: history (L, hh) + attn_mask (L,) + current (1 + N + 2)
        # current = [correct_door, bribe_0..N-1, host_cumulative_profit,
        #            round_normalised]
        host_current_size = 1 + self.num_players + 2
        host_obs_size = L * hh + L + host_current_size  # flattened

        # action: [public_signal_frac, priv_sig_0_frac, …, priv_sig_{N-1}_frac]
        host_act_size = 1 + self.num_players  # all in [0,1] → door index

        # ------ PLAYER ------
        # observation: history (L, ph) + attn_mask (L,) + current (5)
        # current = [balance, public_signal, private_signal, round_normalised,
        #            is_bankrupt]
        player_current_size = 5
        player_obs_size = L * ph + L + player_current_size

        # Phase I action: [bribe_fraction]  ∈ [0,1]
        # Phase II action: [door_fraction, bet_fraction]  ∈ [0,1]²
        # We use the larger Phase II space for both; Phase I only reads index 0.
        player_act_size = 2

        self.observation_spaces: dict[str, spaces.Space] = {}
        self.action_spaces: dict[str, spaces.Space] = {}

        self.observation_spaces["host"] = spaces.Box(
            low=PAD_VALUE,
            high=np.inf,
            shape=(host_obs_size,),
            dtype=np.float32,
        )
        self.action_spaces["host"] = spaces.Box(
            low=0.0, high=1.0, shape=(host_act_size,), dtype=np.float32
        )

        for pid in range(self.num_players):
            name = f"player_{pid}"
            self.observation_spaces[name] = spaces.Box(
                low=PAD_VALUE,
                high=np.inf,
                shape=(player_obs_size,),
                dtype=np.float32,
            )
            self.action_spaces[name] = spaces.Box(
                low=0.0, high=1.0, shape=(player_act_size,), dtype=np.float32
            )

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self._phase = 0
        self._round = 0
        self._host_cumulative_profit = 0.0

        self._balances = {a: self.initial_balance for a in self.agents}
        self._bribes = {}
        self._bets = {}
        self._door_choices = {}
        self._private_signals = {}
        self._public_signal = 0

        # Snapshot of the most recently settled round (populated after each step_all)
        self.last_round_info: dict = {}

        self._player_hist = {
            f"player_{i}": collections.deque(maxlen=self.history_window)
            for i in range(self.num_players)
        }
        self._host_hist = collections.deque(maxlen=self.history_window)

        # Draw correct door for round 0
        self._correct_door = int(self._rng.integers(0, self.num_doors))

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        if _HAS_PETTINGZOO:
            self._agent_selector.reinit(self.agents)
            self.agent_selection = self._agent_selector.next()

    # ------------------------------------------------------------------

    def observe(self, agent: str) -> np.ndarray:
        """Return the flat observation vector for *agent*."""
        L = self.history_window
        if agent == "host":
            return self._host_obs()
        else:
            return self._player_obs(agent)

    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> None:
        """
        Process one agent's action.

        The environment collects actions from all agents before advancing
        the game state.  Actions are buffered internally; settlement happens
        when the last agent in each phase has acted.
        """
        agent = self.agent_selection  # type: ignore[attr-defined]
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        if self._phase == 0:
            self._handle_phase1(agent, action)
        else:
            self._handle_phase2(agent, action)

        # Advance selector
        self._agent_selector.next()
        if self._agent_selector.is_last():
            self._end_of_phase()
        self.agent_selection = self._agent_selector.agent_order[
            self._agent_selector._current_agent
        ] if _HAS_PETTINGZOO else agent

    # ------------------------------------------------------------------
    # Simplified non-AEC step (for custom training loops)
    # ------------------------------------------------------------------

    def step_all(
        self,
        actions_phase1: dict[str, float],
        actions_phase2: dict[str, tuple[int, float]],
    ) -> dict[str, float]:
        """
        Convenience method for custom training loops that bypass AEC ordering.

        Parameters
        ----------
        actions_phase1 : {agent_id: bribe_fraction}
            Bribe fraction ∈ [0,1] for each player; ignored for "host".
        actions_phase2 : {agent_id: (door_fraction, bet_fraction)}
            (door_fraction, bet_fraction) ∈ [0,1]² for each player.
            Host is not included (host acts only in Phase I via signals).

        Returns
        -------
        rewards : dict mapping agent_id → scalar reward
        """
        # ----- Phase I: players bribe -----
        self._bribes = {}
        for pid in range(self.num_players):
            name = f"player_{pid}"
            frac = float(np.clip(actions_phase1.get(name, 0.0), 0.0, 1.0))
            bal = self._balances[name]
            bribe = frac * bal
            bribe = min(bribe, bal)
            self._bribes[name] = bribe
            self._balances[name] -= bribe

        # ----- Phase I: host emits signals -----
        host_action = actions_phase1.get("host", np.zeros(1 + self.num_players))
        if np.isscalar(host_action):
            host_action = np.zeros(1 + self.num_players)
        host_action = np.clip(np.asarray(host_action, dtype=np.float32), 0.0, 1.0)
        self._emit_signals(host_action)

        # ----- Phase II: players bet -----
        self._bets = {}
        self._door_choices = {}
        for pid in range(self.num_players):
            name = f"player_{pid}"
            act = actions_phase2.get(name, (0.0, 0.0))
            door_frac, bet_frac = float(act[0]), float(act[1])
            door = int(np.clip(round(door_frac * (self.num_doors - 1)), 0, self.num_doors - 1))
            bal = self._balances[name]
            bet = float(np.clip(bet_frac, 0.0, 1.0)) * bal
            bet = min(bet, bal)
            self._door_choices[name] = door
            self._bets[name] = bet
            self._balances[name] -= bet

        rewards = self._settle()
        self._update_history(rewards)
        self._prepare_next_round()
        return rewards

    # ------------------------------------------------------------------
    # Phase handlers (AEC mode)
    # ------------------------------------------------------------------

    def _handle_phase1(self, agent: str, action: np.ndarray) -> None:
        """Buffer Phase I actions."""
        if agent == "host":
            # Host action is stored; signals emitted when phase ends
            self._pending_host_action = action
        else:
            bal = self._balances[agent]
            bribe = float(action[0]) * bal
            bribe = min(bribe, bal)
            self._bribes[agent] = bribe
            self._balances[agent] -= bribe

    def _handle_phase2(self, agent: str, action: np.ndarray) -> None:
        """Buffer Phase II actions (players only)."""
        if agent == "host":
            return  # host does not act in Phase II
        bal = self._balances[agent]
        door = int(np.clip(round(float(action[0]) * (self.num_doors - 1)), 0, self.num_doors - 1))
        bet = float(action[1]) * bal
        bet = min(bet, bal)
        self._door_choices[agent] = door
        self._bets[agent] = bet
        self._balances[agent] -= bet

    def _end_of_phase(self) -> None:
        if self._phase == 0:
            # Emit host signals, switch to Phase II
            host_act = getattr(self, "_pending_host_action", np.zeros(1 + self.num_players))
            self._emit_signals(host_act)
            self._phase = 1
            if _HAS_PETTINGZOO:
                # Remove host from Phase II selector (host doesn't act)
                phase2_agents = [a for a in self.agents if a != "host"]
                self._agent_selector.reinit(phase2_agents)
        else:
            # Settle, update history, start next round
            rewards = self._settle()
            self._update_history(rewards)
            for a, r in rewards.items():
                self.rewards[a] = r
                self._cumulative_rewards[a] += r
            self._prepare_next_round()

    # ------------------------------------------------------------------
    # Core mechanics
    # ------------------------------------------------------------------

    def _emit_signals(self, host_action: np.ndarray) -> None:
        """Convert host action fractions → signal door indices."""
        self._public_signal = int(
            np.clip(round(float(host_action[0]) * (self.num_doors - 1)), 0, self.num_doors - 1)
        )
        for i in range(self.num_players):
            name = f"player_{i}"
            frac = float(host_action[1 + i]) if len(host_action) > 1 + i else float(host_action[0])
            door = int(np.clip(round(frac * (self.num_doors - 1)), 0, self.num_doors - 1))
            self._private_signals[name] = door

    def calculate_payout(self, individual_bet: float, x: float) -> float:
        """
        Payout for a single winning bet given winning ratio x.

            Payout = b * (1 + (1 - θ) / x)

        where θ = payout_threshold.
        """
        if x <= 0:
            return 0.0
        multiplier = 1.0 + (1.0 - self.payout_threshold) / x
        return individual_bet * multiplier

    def _settle(self) -> dict[str, float]:
        """Compute rewards after Phase II; update balances."""
        rewards: dict[str, float] = {a: 0.0 for a in self.agents}

        total_pool = sum(self._bets.values())
        total_bribe = sum(self._bribes.values())

        if total_pool <= 0:
            # No bets placed: only bribes collected by host
            host_reward = total_bribe
            self._host_cumulative_profit += host_reward
            rewards["host"] = host_reward
            return rewards

        # Winning volume
        winning_bets: dict[str, float] = {
            name: bet
            for name, bet in self._bets.items()
            if self._door_choices.get(name) == self._correct_door
        }
        total_winning = sum(winning_bets.values())
        x = total_winning / total_pool if total_pool > 0 else 0.0

        # Distribute payouts to winners
        total_payout = 0.0
        for name, bet in winning_bets.items():
            payout = self.calculate_payout(bet, x) if x > 0 else 0.0
            self._balances[name] += payout
            total_payout += payout
            # Player reward = net gain: payout - (bet already deducted) - bribe
            rewards[name] = payout - bet - self._bribes.get(name, 0.0)

        # Losers' reward = -(bet + bribe)
        for name in self._bets:
            if name not in winning_bets:
                rewards[name] = -(self._bets[name] + self._bribes.get(name, 0.0))

        # Fee rake for host
        fee = self.fee_rate * total_pool
        self._balances["host"] = self._balances.get("host", 0.0) + fee

        # Host reward = pool - payouts + bribes + fee_rake
        host_profit = total_pool - total_payout + total_bribe + fee
        # Subtract the fee that was already counted in pool vs payout gap
        # Simplified: host collects loser bets + bribes + fee, pays out winners
        host_reward = (total_pool - total_winning) + total_bribe + fee - (total_payout - total_winning)
        self._host_cumulative_profit += host_reward
        rewards["host"] = host_reward

        return rewards

    def _welfare_check(self) -> None:
        """Ensure every player has at least welfare_amount after settlement.

        Using `< welfare_amount` (not just `<= 0`) guarantees a meaningful
        minimum floor so players can always participate in the next round and
        Transformer history never contains degenerate zero-balance rows.
        """
        for pid in range(self.num_players):
            name = f"player_{pid}"
            if self._balances[name] < self.welfare_amount:
                self._balances[name] = self.welfare_amount

    def _prepare_next_round(self) -> None:
        """Welfare check, draw new correct door, increment round counter."""
        self._welfare_check()
        self._round += 1
        self._correct_door = int(self._rng.integers(0, self.num_doors))
        self._phase = 0
        self._bribes = {}
        self._bets = {}
        self._door_choices = {}
        self._private_signals = {}

        # Check episode termination
        if self.max_rounds > 0 and self._round >= self.max_rounds:
            for a in self.agents:
                self.truncations[a] = True

        if _HAS_PETTINGZOO:
            all_agents = list(self.possible_agents)
            self._agent_selector.reinit(all_agents)

    # ------------------------------------------------------------------
    # History bookkeeping
    # ------------------------------------------------------------------

    def _update_history(self, rewards: dict[str, float]) -> None:
        """Push current round data into history deques."""
        total_pool = sum(self._bets.values()) if self._bets else 1e-8
        door_ratios = np.zeros(self.num_doors, dtype=np.float32)
        for name, bet in self._bets.items():
            d = self._door_choices.get(name, 0)
            door_ratios[d] += bet
        if total_pool > 0:
            door_ratios /= total_pool

        for pid in range(self.num_players):
            name = f"player_{pid}"
            bal_before_round = self._balances[name]
            won = int(self._door_choices.get(name, -1) == self._correct_door)
            payout = max(rewards.get(name, 0.0), 0.0)
            feat = np.concatenate([
                [self._door_choices.get(name, PAD_VALUE)],   # door_choice
                [self._private_signals.get(name, PAD_VALUE)],# private_signal
                [self._public_signal],                         # public_signal
                [self._bribes.get(name, 0.0)],                # bribe
                [self._bets.get(name, 0.0)],                  # bet
                [bal_before_round],                            # balance
                [won],                                         # won
                [payout],                                      # payout
                door_ratios,                                   # door ratios
            ], dtype=np.float32)
            self._player_hist[name].append(feat)

        # Snapshot everything before _prepare_next_round clears the dicts
        self.last_round_info = {
            "round":           self._round,
            "correct_door":    self._correct_door,
            "public_signal":   self._public_signal,
            "private_signals": dict(self._private_signals),
            "bribes":          dict(self._bribes),
            "bets":            dict(self._bets),
            "door_choices":    dict(self._door_choices),
            "balances":        {k: v for k, v in self._balances.items()
                                if k != "host"},
            "door_ratios":     door_ratios.tolist(),
            "rewards":         dict(rewards),
        }

        # Host history
        player_fields = []
        for pid in range(self.num_players):
            name = f"player_{pid}"
            player_fields.extend([
                self._bribes.get(name, 0.0),
                self._private_signals.get(name, PAD_VALUE),
                self._balances[name],
                self._door_choices.get(name, PAD_VALUE),
                float(self._door_choices.get(name, -1) == self._correct_door),
            ])
        host_feat = np.array(
            [self._correct_door, self._public_signal] + player_fields,
            dtype=np.float32,
        )
        self._host_hist.append(host_feat)

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _player_obs(self, agent: str) -> np.ndarray:
        L = self.history_window
        ph = self._player_hist_feat
        hist = _pad_history(self._player_hist[agent], L, ph)
        mask = _build_attention_mask(self._player_hist[agent], L)
        current = np.array([
            self._balances[agent] / self.initial_balance,          # normalised balance
            self._public_signal / (self.num_doors - 1),            # public signal
            self._private_signals.get(agent, PAD_VALUE) / (self.num_doors - 1),
            self._round / max(self.max_rounds, 1),                 # round progress
            float(self._balances[agent] <= self.welfare_amount),   # is_bankrupt flag
        ], dtype=np.float32)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    def _host_obs(self) -> np.ndarray:
        L = self.history_window
        hh = self._host_hist_feat
        hist = _pad_history(self._host_hist, L, hh)
        mask = _build_attention_mask(self._host_hist, L)
        bribes = [self._bribes.get(f"player_{i}", 0.0) for i in range(self.num_players)]
        current = np.array(
            [self._correct_door / (self.num_doors - 1)]
            + [b / self.initial_balance for b in bribes]
            + [self._host_cumulative_profit / (self.initial_balance * self.num_players),
               self._round / max(self.max_rounds, 1)],
            dtype=np.float32,
        )
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    # ------------------------------------------------------------------
    # Rendering (minimal)
    # ------------------------------------------------------------------

    def render(self) -> None:
        print(
            f"Round {self._round:4d} | Phase {self._phase} | "
            f"Correct door: {self._correct_door} | "
            f"Host profit: {self._host_cumulative_profit:.2f}"
        )
        for pid in range(self.num_players):
            name = f"player_{pid}"
            print(
                f"  {name}: balance={self._balances[name]:.2f}  "
                f"door={self._door_choices.get(name, '-')}  "
                f"bet={self._bets.get(name, 0):.2f}  "
                f"bribe={self._bribes.get(name, 0):.2f}"
            )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # PettingZoo compatibility helpers
    # ------------------------------------------------------------------

    def _was_dead_step(self, action: np.ndarray) -> None:
        """Handle step called for a terminated/truncated agent."""
        if _HAS_PETTINGZOO:
            # Standard PettingZoo dead-step behaviour
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.agent_order[
                self._agent_selector._current_agent
            ]

    @functools.lru_cache(maxsize=None)
    def _observation_space(self, agent: str) -> spaces.Space:  # noqa: D401
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def _action_space(self, agent: str) -> spaces.Space:  # noqa: D401
        return self.action_spaces[agent]