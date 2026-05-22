"""
OracleGambit (Simplified): Public Signal Manipulation in Multi-Agent RL
=======================================================================
Simplified version focusing on the core strategic tension:
  - Can the Host manipulate Players through a public signal?
  - Can Players learn to distrust a deceptive Host?

Removed from full version: Bribes, Private Signals, Balance/Bankruptcy.

Game flow per round (single phase):
  1. Environment draws the correct door (hidden from Players).
  2. Host observes correct door → outputs a public_signal (door 0~3).
  3. All Players receive public_signal → each picks a door.
  4. Settlement: compute win-ratio x, apply dynamic payout, distribute rewards.

Payout rule (dynamic odds, threshold θ = 0.20):
  - Each Player bets 1 unit (fixed).
  - Win-ratio  x = (# winners) / N
  - Multiplier M(x) = 1 + (1 - θ) / x
  - Winner net reward = M(x) - 1 = (1 - θ) / x
  - Loser  net reward = -1
- Host   net reward = N · (θ - x)   [positive iff x < θ; = N when x = 0]

Zero-sum intuition at threshold:
  x = θ  →  total payout = N  (break-even for host)
  x < θ  →  Host profits; x > θ  →  Host loses.
"""

from __future__ import annotations

import collections
import functools
from typing import Any

import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Optional PettingZoo import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils.agent_selector import agent_selector
    _HAS_PETTINGZOO = True
except ImportError:
    _HAS_PETTINGZOO = False

    class AECEnv:  # type: ignore[no-redef]
        metadata: dict = {}
        def __init__(self) -> None:
            pass

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "num_players":        6,
    "num_doors":          4,
    # Host breaks even when win-ratio == payout_threshold
    "payout_threshold":   0.20,
    # History buffer length (rounds kept in observation)
    "history_window":     50,
    # Episode length (0 = unlimited)
    "max_rounds":         500,
    "seed":               None,
}

PAD_VALUE: float = -1.0   # padding token for empty history slots

# ---------------------------------------------------------------------------
# History feature layout
# ---------------------------------------------------------------------------
# Player history — features per round:
#   door_choice      (1)  : door the player chose, normalised to [0,1]
#   public_signal    (1)  : door the host broadcast, normalised
#   followed_signal  (1)  : 1 if door_choice == public_signal, else 0
#   won              (1)  : 1 if player won, else 0
#   door_ratios      (D)  : fraction of players who picked each door
# Total = 4 + num_doors
_PLAYER_HIST_BASE = 4

def _player_hist_size(num_doors: int) -> int:
    return _PLAYER_HIST_BASE + num_doors

# Host history — features per round:
#   correct_door     (1)  : normalised
#   public_signal    (1)  : normalised
#   signal_honest    (1)  : 1 if signal == correct_door
#   win_ratio        (1)  : x = winners / N
# Total = 4
_HOST_HIST_SIZE = 4

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pad_history(buf: collections.deque, window: int, feat: int) -> np.ndarray:
    """Return (window, feat) float32; older rows padded with PAD_VALUE."""
    out = np.full((window, feat), PAD_VALUE, dtype=np.float32)
    data = list(buf)
    n = len(data)
    if n:
        out[-n:] = np.asarray(data, dtype=np.float32)
    return out


def _attention_mask(buf: collections.deque, window: int) -> np.ndarray:
    """Bool mask (window,): True = valid timestep, False = padding."""
    mask = np.zeros(window, dtype=bool)
    n = len(buf)
    if n:
        mask[-n:] = True
    return mask


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OracleGambitEnv(AECEnv):
    """
    Simplified OracleGambit: public-signal-only, fixed-bet, no balance tracking.

    Agent IDs
    ---------
    ``"host"``        – the Oracle (full information)
    ``"player_0"`` …  – the N Players (observe only public_signal)

    Action spaces (all continuous [0, 1], mapped internally to door index)
    -------------
    host   : Box(1,)  → public_signal = round(a[0] * (D-1))
    player : Box(1,)  → door_choice   = round(a[0] * (D-1))

    Observation spaces (flat float32 vectors)
    ------------------
    player : [history(L, 4+D) | attn_mask(L) | current(D+1)]
    host   : [history(L, 4)   | attn_mask(L) | current(D+1)]

    Rewards
    -------
    winner player : (1 - θ) / x          (net gain, fixed 1-unit stake)
    loser  player : -1
    host          : N * (θ - x)          (positive iff x < θ)

    last_round_info (dict, updated after every step_all)
    ---------------
    Keys: round, correct_door, public_signal, door_choices (dict),
          win_ratio, rewards (dict), door_ratios (list[float])
    """

    metadata = {"render_modes": ["human"], "name": "oracle_gambit_simple_v0"}

    # ------------------------------------------------------------------
    def __init__(self, **cfg_override) -> None:
        super().__init__()

        cfg = {**DEFAULT_CONFIG, **cfg_override}
        self.num_players:      int   = int(cfg["num_players"])
        self.num_doors:        int   = int(cfg["num_doors"])
        self.payout_threshold: float = float(cfg["payout_threshold"])
        self.history_window:   int   = int(cfg["history_window"])
        self.max_rounds:       int   = int(cfg["max_rounds"])
        self._seed                   = cfg["seed"]

        self._rng = np.random.default_rng(self._seed)

        self.possible_agents: list[str] = (
            ["host"] + [f"player_{i}" for i in range(self.num_players)]
        )

        self._player_hist_feat = _player_hist_size(self.num_doors)
        self._host_hist_feat   = _HOST_HIST_SIZE

        self._build_spaces()

        # Runtime state (initialised in reset)
        self.agents:           list[str]        = []
        self._round:           int              = 0
        self._correct_door:    int              = 0
        self._public_signal:   int              = 0
        self._door_choices:    dict[str, int]   = {}

        self._player_hist: dict[str, collections.deque] = {}
        self._host_hist:   collections.deque            = collections.deque(
            maxlen=self.history_window
        )
        self._host_cumulative_reward: float = 0.0

        # Snapshot filled after every settled round
        self.last_round_info: dict = {}

        # PettingZoo bookkeeping
        self.rewards:             dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations:        dict[str, bool]  = {}
        self.truncations:         dict[str, bool]  = {}
        self.infos:               dict[str, dict]  = {}

        if _HAS_PETTINGZOO:
            self._agent_selector = agent_selector(self.possible_agents)

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------

    def _build_spaces(self) -> None:
        L   = self.history_window
        ph  = self._player_hist_feat   # 4 + D
        hh  = self._host_hist_feat     # 4
        D   = self.num_doors

        # current context: one-hot signal (D) + round_fraction (1)
        current_size = D + 1

        player_obs_size = L * ph + L + current_size
        host_obs_size   = L * hh + L + current_size

        self.observation_spaces: dict[str, spaces.Space] = {
            "host": spaces.Box(PAD_VALUE, np.inf, (host_obs_size,), np.float32)
        }
        self.action_spaces: dict[str, spaces.Space] = {
            "host": spaces.Box(0.0, 1.0, (1,), np.float32)
        }
        for i in range(self.num_players):
            name = f"player_{i}"
            self.observation_spaces[name] = spaces.Box(
                PAD_VALUE, np.inf, (player_obs_size,), np.float32
            )
            self.action_spaces[name] = spaces.Box(0.0, 1.0, (1,), np.float32)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents               = list(self.possible_agents)
        self._round               = 0
        self._host_cumulative_reward = 0.0
        self._correct_door        = int(self._rng.integers(0, self.num_doors))
        self._public_signal       = 0
        self._door_choices        = {}
        self.last_round_info      = {}

        self._player_hist = {
            f"player_{i}": collections.deque(maxlen=self.history_window)
            for i in range(self.num_players)
        }
        self._host_hist = collections.deque(maxlen=self.history_window)

        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {} for a in self.agents}

        if _HAS_PETTINGZOO:
            self._agent_selector.reinit(self.agents)
            self.agent_selection = self._agent_selector.next()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def observe(self, agent: str) -> np.ndarray:
        if agent == "host":
            return self._host_obs()
        return self._player_obs(agent)

    def _player_obs(self, agent: str) -> np.ndarray:
        L   = self.history_window
        ph  = self._player_hist_feat
        D   = self.num_doors
        hist = _pad_history(self._player_hist[agent], L, ph)
        mask = _attention_mask(self._player_hist[agent], L)
        sig_onehot = np.zeros(D, dtype=np.float32)
        sig_onehot[self._public_signal] = 1.0
        round_frac = min(1.0, self._round / self.max_rounds) if self.max_rounds > 0 else 0.0
        current = np.append(sig_onehot, round_frac)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    def _host_obs(self) -> np.ndarray:
        L   = self.history_window
        hh  = self._host_hist_feat
        D   = self.num_doors
        hist = _pad_history(self._host_hist, L, hh)
        mask = _attention_mask(self._host_hist, L)
        door_onehot = np.zeros(D, dtype=np.float32)
        door_onehot[self._correct_door] = 1.0
        round_frac = min(1.0, self._round / self.max_rounds) if self.max_rounds > 0 else 0.0
        current = np.append(door_onehot, round_frac)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    # ------------------------------------------------------------------
    # Core payout math
    # ------------------------------------------------------------------

    def calculate_multiplier(self, x: float) -> float:
        """
        Dynamic payout multiplier given winning ratio x.

            M(x) = 1 + (1 - θ) / x

        At x = θ the total payout equals the pool (break-even for host).
        Returns 0.0 if x <= 0 (no winners → no payout issued).
        """
        if x <= 0.0:
            return 0.0
        return 1.0 + (1.0 - self.payout_threshold) / x

    def _settle(self) -> dict[str, float]:
        """Compute rewards from current _door_choices and _correct_door."""
        N = self.num_players
        θ = self.payout_threshold

        winners = [
            name for name, door in self._door_choices.items()
            if door == self._correct_door
        ]
        W = len(winners)
        x = W / N if N > 0 else 0.0
        M = self.calculate_multiplier(x)

        rewards: dict[str, float] = {}
        for pid in range(N):
            name = f"player_{pid}"
            if name in winners:
                rewards[name] = M - 1.0   # net gain after recovering stake
            else:
                rewards[name] = -1.0

        # Host profit = pool − total payout.  Works for W=0 edge case too:
        #   W>0: N - W*M = N - N*x*(1+(1-θ)/x) = N*(θ-x)
        #   W=0: N - 0   = N  (host keeps everything)
        rewards["host"] = float(N) - float(W) * M
        self._host_cumulative_reward += rewards["host"]
        return rewards

    # ------------------------------------------------------------------
    # History update
    # ------------------------------------------------------------------

    def _update_history(self, rewards: dict[str, float]) -> None:
        N    = self.num_players
        D    = self.num_doors
        norm = D - 1 if D > 1 else 1

        # Door-choice distribution
        door_counts = np.zeros(D, dtype=np.float32)
        for door in self._door_choices.values():
            door_counts[door] += 1
        door_ratios = door_counts / N if N > 0 else door_counts

        # Per-player history entry
        for pid in range(N):
            name     = f"player_{pid}"
            door     = self._door_choices.get(name, 0)
            won      = int(door == self._correct_door)
            followed = int(door == self._public_signal)
            feat = np.array([
                door / norm,
                self._public_signal / norm,
                float(followed),
                float(won),
                *door_ratios,
            ], dtype=np.float32)
            self._player_hist[name].append(feat)

        # Host history entry
        W = len([n for n, d in self._door_choices.items() if d == self._correct_door])
        x = W / N if N > 0 else 0.0
        host_feat = np.array([
            self._correct_door / norm,
            self._public_signal / norm,
            float(self._correct_door == self._public_signal),
            x,
        ], dtype=np.float32)
        self._host_hist.append(host_feat)

        # Snapshot for external inspection
        self.last_round_info = {
            "round":         self._round,
            "correct_door":  self._correct_door,
            "public_signal": self._public_signal,
            "door_choices":  dict(self._door_choices),
            "door_ratios":   door_ratios.tolist(),
            "win_ratio":     x,
            "rewards":       dict(rewards),
        }

    # ------------------------------------------------------------------
    # Round advancement
    # ------------------------------------------------------------------

    def _prepare_next_round(self) -> None:
        self._round       += 1
        self._correct_door = int(self._rng.integers(0, self.num_doors))
        self._public_signal = 0
        self._door_choices  = {}

        if self.max_rounds > 0 and self._round >= self.max_rounds:
            for a in self.agents:
                self.truncations[a] = True

        if _HAS_PETTINGZOO:
            self._agent_selector.reinit(self.agents)

    # ------------------------------------------------------------------
    # Primary training interface: step_all()
    # ------------------------------------------------------------------

    def step_all(
        self,
        host_action:    np.ndarray | float,
        player_actions: dict[str, np.ndarray | float],
    ) -> dict[str, float]:
        """
        Run one complete round.

        Parameters
        ----------
        host_action : scalar or array-like in [0, 1]
            Host's public_signal as a fraction → mapped to door index.
        player_actions : {player_id: scalar or array-like in [0, 1]}
            Each player's door choice as a fraction → mapped to door index.

        Returns
        -------
        rewards : dict[agent_id, float]
        """
        norm = self.num_doors - 1 if self.num_doors > 1 else 1

        # Host emits public signal
        h = float(np.clip(float(np.asarray(host_action).flat[0]), 0.0, 1.0))
        self._public_signal = int(round(h * norm))

        # Players choose doors
        self._door_choices = {}
        for pid in range(self.num_players):
            name = f"player_{pid}"
            p = float(np.clip(float(np.asarray(player_actions.get(name, 0.0)).flat[0]), 0.0, 1.0))
            self._door_choices[name] = int(round(p * norm))

        rewards = self._settle()
        self._update_history(rewards)
        self._prepare_next_round()
        return rewards

    # ------------------------------------------------------------------
    # PettingZoo AEC step() (turn-based: host first, then players)
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> None:
        agent = self.agent_selection  # type: ignore[attr-defined]
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        norm = self.num_doors - 1 if self.num_doors > 1 else 1
        a    = float(np.clip(float(np.asarray(action).flat[0]), 0.0, 1.0))
        door = int(round(a * norm))

        if agent == "host":
            self._public_signal = door
        else:
            self._door_choices[agent] = door

        if _HAS_PETTINGZOO:
            self._agent_selector.next()
            if self._agent_selector.is_last():
                rewards = self._settle()
                self._update_history(rewards)
                for a_id, r in rewards.items():
                    self.rewards[a_id]             = r
                    self._cumulative_rewards[a_id] += r
                self._prepare_next_round()
            self.agent_selection = self._agent_selector.agent_order[
                self._agent_selector._current_agent
            ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        info = self.last_round_info
        if not info:
            print(f"Round {self._round} — no data yet.")
            return
        D      = self.num_doors
        labels = list("ABCD")[:D]
        print(
            f"Round {info['round']:>4}  "
            f"Correct={labels[info['correct_door']]}  "
            f"Signal={labels[info['public_signal']]}"
            f"{'✓' if info['correct_door'] == info['public_signal'] else '✗'}  "
            f"x={info['win_ratio']:.3f}  "
            f"HostRwd={info['rewards'].get('host', 0):+.2f}"
        )
        for pid in range(self.num_players):
            name = f"player_{pid}"
            door = info["door_choices"].get(name, -1)
            rwd  = info["rewards"].get(name, 0.0)
            won  = door == info["correct_door"]
            print(
                f"  {name}: door={labels[door] if door >= 0 else '?'}"
                f"{'✓' if won else ' '}  rwd={rwd:+.3f}"
            )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # PettingZoo helpers
    # ------------------------------------------------------------------

    def _was_dead_step(self, action: np.ndarray) -> None:
        if _HAS_PETTINGZOO:
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.agent_order[
                self._agent_selector._current_agent
            ]

    @functools.lru_cache(maxsize=None)
    def _observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def _action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]
