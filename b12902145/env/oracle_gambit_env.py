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
    # Betting system: discrete fraction levels of current balance
    "num_bet_fracs":      5,
    "bet_fracs":          [0.05, 0.10, 0.25, 0.50, 1.00],
    # Player balance & bankruptcy
    "initial_capital_min":  50.0,
    "initial_capital_max":  150.0,
    "bankruptcy_threshold": 10.0,
    "bankruptcy_penalty":    0.0,   # 0 = no artificial penalty; losing the bet is punishment enough
    # Bet-encouragement shaping: bonus = max(0, win_rate-0.25) * bet_frac_norm * scale
    # Rewards players for betting more when their recent win-rate exceeds random baseline.
    # Set to 0 to disable.
    "bet_shaping_scale":  2.0,
    "seed":               None,
}

PAD_VALUE: float = -1.0   # padding token for empty history slots

# ---------------------------------------------------------------------------
# History feature layout
# ---------------------------------------------------------------------------
# Player history — features per round:
#   bet_frac_norm    (1)  : chosen bet-fraction index normalised to [0,1]
#   door_choice      (1)  : door the player chose, normalised to [0,1]
#   public_signal    (1)  : door the host broadcast, normalised
#   followed_signal  (1)  : 1 if door_choice == public_signal, else 0
#   won              (1)  : 1 if player won, else 0  ← index 4
#   door_ratios      (D)  : fraction of players who picked each door
# Total = 5 + num_doors
_PLAYER_HIST_BASE = 5

def _player_hist_size(num_doors: int) -> int:
    return _PLAYER_HIST_BASE + num_doors

# Host history — features per round:
#   correct_door     (1)  : normalised
#   public_signal    (1)  : normalised
#   signal_honest    (1)  : 1 if signal == correct_door
#   win_ratio        (1)  : x = winners / N
#   avg_bet_norm     (1)  : mean(b_i) normalised to [0,1] — player confidence signal
# Total = 5
_HOST_HIST_SIZE = 5

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
        self.num_bet_fracs:    int   = int(cfg["num_bet_fracs"])
        self.bet_fracs:        list  = list(cfg["bet_fracs"])
        self.initial_capital_min:  float = float(cfg["initial_capital_min"])
        self.initial_capital_max:  float = float(cfg["initial_capital_max"])
        self.bankruptcy_threshold: float = float(cfg["bankruptcy_threshold"])
        self.bankruptcy_penalty:   float = float(cfg["bankruptcy_penalty"])
        self.bet_shaping_scale:    float = float(cfg["bet_shaping_scale"])
        self._seed                       = cfg["seed"]

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
        self._player_bets:            dict[str, int]   = {}  # frac_idx 0..NF-1 per player
        self._player_balance:         dict[str, float] = {}
        self._player_initial_capital: dict[str, float] = {}

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
        ph  = self._player_hist_feat   # 6 + D
        hh  = self._host_hist_feat     # 5
        D   = self.num_doors
        NF  = self.num_bet_fracs

        # Pre-signal obs (bet phase): history + mask + [round_frac(1)]
        pre_signal_obs_size = L * ph + L + 1

        # Post-signal obs (door phase): history + mask + [signal_onehot(D), bet_frac_norm(1), round_frac(1)]
        door_obs_size = L * ph + L + D + 2

        # Host obs (post-bets): history + mask + [correct_door_onehot(D), bet_frac_hist(NF), round_frac(1)]
        host_obs_size = L * hh + L + D + NF + 1

        # Expose sizes for external agent construction
        self.pre_signal_obs_size = pre_signal_obs_size
        self.door_obs_size       = door_obs_size
        self.host_obs_size       = host_obs_size

        self.observation_spaces: dict[str, spaces.Space] = {
            "host": spaces.Box(PAD_VALUE, np.inf, (host_obs_size,), np.float32)
        }
        self.action_spaces: dict[str, spaces.Space] = {
            "host": spaces.Box(0.0, 1.0, (1,), np.float32)
        }
        for i in range(self.num_players):
            name = f"player_{i}"
            # observation_space() returns the *door* (post-signal) obs by default
            self.observation_spaces[name] = spaces.Box(
                PAD_VALUE, np.inf, (door_obs_size,), np.float32
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
        self._player_bets         = {}
        self._player_balance        = {}
        self._player_initial_capital = {}
        self.last_round_info      = {}

        self._player_hist = {
            f"player_{i}": collections.deque(maxlen=self.history_window)
            for i in range(self.num_players)
        }
        self._host_hist = collections.deque(maxlen=self.history_window)

        # Initialize player balances (random from U[min, max])
        for i in range(self.num_players):
            name = f"player_{i}"
            cap = float(self._rng.uniform(
                self.initial_capital_min, self.initial_capital_max
            ))
            self._player_initial_capital[name] = cap
            self._player_balance[name]          = cap

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

    def observe_pre_signal(self, agent: str) -> np.ndarray:
        """
        Return a player’s *pre-signal* observation for the bet phase.
        No signal or bet context is included; only history and round fraction.
        """
        if agent == "host":
            raise ValueError("Host has no pre-signal observation.")
        return self._player_pre_signal_obs(agent)

    def _player_pre_signal_obs(self, agent: str) -> np.ndarray:
        """Player obs before signal is broadcast (used by bet_agent)."""
        L  = self.history_window
        ph = self._player_hist_feat
        hist = _pad_history(self._player_hist[agent], L, ph)
        mask = _attention_mask(self._player_hist[agent], L)
        round_frac = min(1.0, self._round / self.max_rounds) if self.max_rounds > 0 else 0.0
        current = np.array([round_frac], dtype=np.float32)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    def _player_obs(self, agent: str) -> np.ndarray:
        """Post-signal observation (used by door_agent)."""
        L   = self.history_window
        ph  = self._player_hist_feat
        D   = self.num_doors
        NF  = self.num_bet_fracs
        hist = _pad_history(self._player_hist[agent], L, ph)
        mask = _attention_mask(self._player_hist[agent], L)
        sig_onehot = np.zeros(D, dtype=np.float32)
        sig_onehot[self._public_signal] = 1.0
        # Normalised bet fraction index
        frac_idx = self._player_bets.get(agent, 0)
        bet_frac_norm = frac_idx / max(1, NF - 1)
        round_frac = min(1.0, self._round / self.max_rounds) if self.max_rounds > 0 else 0.0
        current = np.array([*sig_onehot, bet_frac_norm, round_frac], dtype=np.float32)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    def _host_obs(self) -> np.ndarray:
        """Host observation: history + correct-door + bet-fraction histogram."""
        L   = self.history_window
        hh  = self._host_hist_feat
        D   = self.num_doors
        NF  = self.num_bet_fracs
        hist = _pad_history(self._host_hist, L, hh)
        mask = _attention_mask(self._host_hist, L)
        door_onehot = np.zeros(D, dtype=np.float32)
        door_onehot[self._correct_door] = 1.0
        # Bet fraction histogram: fraction of players at each frac-index level
        bet_frac_hist = np.zeros(NF, dtype=np.float32)
        if self._player_bets:
            for fv in self._player_bets.values():
                bet_frac_hist[fv] += 1
            bet_frac_hist /= len(self._player_bets)
        round_frac = min(1.0, self._round / self.max_rounds) if self.max_rounds > 0 else 0.0
        current = np.array([*door_onehot, *bet_frac_hist, round_frac], dtype=np.float32)
        return np.concatenate([hist.flatten(), mask.astype(np.float32), current])

    # ------------------------------------------------------------------
    # Core payout math
    # ------------------------------------------------------------------

    def calculate_multiplier(self, x: float) -> float:
        """
        Dynamic payout multiplier given winning ratio x  (only used when x > 0).

            M(x) = 1 + (1 - θ) / x

        At x = θ the total payout to winners equals the pool (host breaks even).
        When x = 0 (no winners), the host keeps the entire pool B_all — see _settle().
        Returns 0.0 if x <= 0 (caller should use the W=0 branch instead).
        """
        if x <= 0.0:
            return 0.0
        return 1.0 + (1.0 - self.payout_threshold) / x

    def _settle(self) -> dict[str, float]:
        """Compute rewards from current _door_choices, _player_bets, _player_balance."""
        N = self.num_players
        θ = self.payout_threshold
        threshold = self.bankruptcy_threshold
        penalty   = self.bankruptcy_penalty

        winners = [
            name for name, door in self._door_choices.items()
            if door == self._correct_door
        ]
        W = len(winners)
        x = W / N if N > 0 else 0.0
        M = self.calculate_multiplier(x)

        # Actual bet amounts = current balance × chosen fraction
        actual_bets: dict[str, float] = {}
        for i in range(N):
            name = f"player_{i}"
            bal  = self._player_balance.get(name, 100.0)
            idx  = self._player_bets.get(name, 0)     # 0 = 5% (safest default)
            frac = self.bet_fracs[min(idx, len(self.bet_fracs) - 1)]
            actual_bets[name] = bal * frac

        B_all = sum(actual_bets.values())
        B_w   = sum(actual_bets[w] for w in winners)

        rewards: dict[str, float] = {}
        for i in range(N):
            name = f"player_{i}"
            b    = actual_bets[name]
            if name in winners:
                net_gain = b * (M - 1.0)
                rewards[name] = net_gain
                self._player_balance[name] = self._player_balance.get(name, 100.0) + net_gain
            else:
                rewards[name] = -b
                self._player_balance[name] = self._player_balance.get(name, 100.0) - b

            # Bankruptcy: refill to threshold and apply penalty
            if self._player_balance[name] < threshold:
                rewards[name] += penalty
                self._player_balance[name] = threshold

        # Host: pool minus winner payouts.
        # W=0 → no winners → host keeps the entire pool (true zero-sum).
        # W>0 → host = collected - paid_out, zero-sum holds exactly.
        if W == 0:
            rewards["host"] = B_all
        else:
            rewards["host"] = B_all - B_w * M
        self._host_cumulative_reward += rewards["host"]

        # Bet-encouragement shaping: bonus = max(0, win_rate - 0.25) * bet_frac_norm * scale
        # Incentivises players to bet more when their recent win-rate exceeds the random baseline.
        # 'won' is at feature index 4 in the player history vector.
        if self.bet_shaping_scale > 0.0:
            NF = self.num_bet_fracs
            _WON_IDX = 4
            for i in range(N):
                name = f"player_{i}"
                hist = self._player_hist[name]
                if len(hist) >= 4:
                    recent_win_rate = float(np.mean([h[_WON_IDX] for h in hist]))
                    frac_idx = self._player_bets.get(name, 0)
                    bet_frac_norm = frac_idx / max(1, NF - 1)
                    shaping = max(0.0, recent_win_rate - 0.25) * bet_frac_norm * self.bet_shaping_scale
                    rewards[name] = rewards.get(name, 0.0) + shaping

        return rewards

    # ------------------------------------------------------------------
    # History update
    # ------------------------------------------------------------------

    def _update_history(self, rewards: dict[str, float]) -> None:
        N    = self.num_players
        D    = self.num_doors
        NF   = self.num_bet_fracs
        norm = D - 1 if D > 1 else 1
        frac_norm_denom = max(1, NF - 1)

        # Door-choice distribution
        door_counts = np.zeros(D, dtype=np.float32)
        for door in self._door_choices.values():
            door_counts[door] += 1
        door_ratios = door_counts / N if N > 0 else door_counts

        # Per-player history entry (bet_frac_norm + 4 scalars + door_ratios)
        # Feature indices: 0=bet_frac_norm, 1=door, 2=signal, 3=followed, 4=won, 5..=door_ratios
        for pid in range(N):
            name     = f"player_{pid}"
            door     = self._door_choices.get(name, 0)
            won      = int(door == self._correct_door)
            followed = int(door == self._public_signal)
            frac_idx = self._player_bets.get(name, 0)
            bet_frac_norm = frac_idx / frac_norm_denom
            feat = np.array([
                bet_frac_norm,
                door / norm,
                self._public_signal / norm,
                float(followed),
                float(won),
                *door_ratios,
            ], dtype=np.float32)
            self._player_hist[name].append(feat)

        # Host history entry (avg_bet_frac_norm instead of avg_bet_norm)
        W = len([n for n, d in self._door_choices.items() if d == self._correct_door])
        x = W / N if N > 0 else 0.0
        all_frac_idxs = [self._player_bets.get(f"player_{i}", 0) for i in range(N)]
        avg_bet_frac_norm = float(np.mean(all_frac_idxs)) / frac_norm_denom
        host_feat = np.array([
            self._correct_door / norm,
            self._public_signal / norm,
            float(self._correct_door == self._public_signal),
            x,
            avg_bet_frac_norm,
        ], dtype=np.float32)
        self._host_hist.append(host_feat)

        # Snapshot for external inspection
        self.last_round_info = {
            "round":             self._round,
            "correct_door":      self._correct_door,
            "public_signal":     self._public_signal,
            "door_choices":      dict(self._door_choices),
            "door_ratios":       door_ratios.tolist(),
            "win_ratio":         x,
            "rewards":           dict(rewards),
            "player_bets":       dict(self._player_bets),   # frac_idx per player
            "avg_bet_frac_idx":  float(np.mean(all_frac_idxs)),
            "player_balances":   dict(self._player_balance),
            "avg_balance":       float(np.mean([
                self._player_balance.get(f"player_{i}", 100.0)
                for i in range(N)
            ])),
        }

    # ------------------------------------------------------------------
    # Round advancement
    # ------------------------------------------------------------------

    def _prepare_next_round(self) -> None:
        self._round       += 1
        self._correct_door = int(self._rng.integers(0, self.num_doors))
        self._public_signal = 0
        self._door_choices  = {}
        self._player_bets   = {}

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
        player_bets:  dict[str, int],
        host_action:  np.ndarray | float,
        player_doors: dict[str, np.ndarray | float],
    ) -> dict[str, float]:
        """
        Run one complete round with variable betting.

        Parameters
        ----------
        player_bets  : {player_id: frac_idx}  integer in [0, num_bet_fracs-1]
        host_action  : scalar or array-like in [0, 1] → public_signal door index
        player_doors : {player_id: scalar or array-like in [0, 1]} → door choice

        Returns
        -------
        rewards : dict[agent_id, float]
        """
        norm = self.num_doors - 1 if self.num_doors > 1 else 1

        # Store & validate bet fraction indices (clamp to [0, NF-1])
        self._player_bets = {
            name: max(0, min(self.num_bet_fracs - 1, int(b)))
            for name, b in player_bets.items()
        }

        # Host emits public signal
        h = float(np.clip(float(np.asarray(host_action).flat[0]), 0.0, 1.0))
        self._public_signal = int(round(h * norm))

        # Players choose doors
        self._door_choices = {}
        for pid in range(self.num_players):
            name = f"player_{pid}"
            p = float(np.clip(float(np.asarray(player_doors.get(name, 0.0)).flat[0]), 0.0, 1.0))
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
            frac_idx = info.get("player_bets", {}).get(name, 0)
            frac_pct = self.bet_fracs[min(frac_idx, len(self.bet_fracs)-1)]
            bal  = info.get("player_balances", {}).get(name, float("nan"))
            rwd  = info["rewards"].get(name, 0.0)
            won  = door == info["correct_door"]
            tick = "\u2713" if won else " "
            print(
                f"  {name}: door={labels[door] if door >= 0 else '?'}"
                f"{tick}  bet={frac_pct:.0%}  bal={bal:.1f}  rwd={rwd:+.3f}"
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
