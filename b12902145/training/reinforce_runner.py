"""
REINFORCE with Moving-Average Baseline for OracleGambit (Phase 1, MLP).

Algorithm overview
------------------
For each batch of B rounds:
  1. Two-phase collection:
       a. Host observes → acts (sets public_signal)
       b. Players observe (now see the signal) → each acts
       c. env.step_all() → rewards
  2. REINFORCE update per role:
       loss = -E[(r - baseline) · log π(a|o)] - β · H[π]
  3. Moving-average baseline update:
       b ← α·b + (1-α)·mean(batch rewards)

Key design decisions
--------------------
* On-policy — no replay buffer.  Each batch is collected fresh, used once, discarded.
* Parameter sharing — all N players share one MlpAgent; the host has its own.
  This means each update uses B·N player samples, greatly improving efficiency.
* Two-phase observation — we manually set env._public_signal = host_door before
  players call env.observe(), so they condition on the current signal (not the
  previous round's default 0).  env.step_all() then re-derives the same value
  from host_frac, so there is no inconsistency.
* Gradient clipping (L2 norm, threshold 1.0) for training stability.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

from agents.mlp_agent import MlpAgent


class ReinforceRunner:
    def __init__(
        self,
        env,
        host_agent: MlpAgent,
        player_agent: MlpAgent,
        lr_host: float = 3e-4,
        lr_player: float = 3e-4,
        entropy_coeff: float = 0.01,
        baseline_momentum: float = 0.99,
        grad_clip: float = 1.0,
    ) -> None:
        self.env = env
        self.host_agent = host_agent
        self.player_agent = player_agent
        self.entropy_coeff = entropy_coeff
        self.baseline_momentum = baseline_momentum
        self.grad_clip = grad_clip

        self.host_opt = optim.Adam(host_agent.parameters(), lr=lr_host)
        self.player_opt = optim.Adam(player_agent.parameters(), lr=lr_player)

        # Moving-average baselines — one scalar per agent role
        self.host_baseline: float = 0.0
        self.player_baseline: float = 0.0

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def _collect_batch(self, batch_size: int) -> dict:
        """
        Run `batch_size` rounds and return collected experience.

        Returns
        -------
        batch : {
            "host":   {"obs": (B, obs_dim), "action": (B,), "reward": (B,)},
            "player": {"obs": (B*N, obs_dim), "action": (B*N,), "reward": (B*N,)},
            "metrics": list[dict]   — env.last_round_info snapshots
        }
        """
        env = self.env
        N = env.num_players
        D = env.num_doors
        norm = D - 1 if D > 1 else 1

        host_obs_l, host_act_l, host_rew_l = [], [], []
        player_obs_l, player_act_l, player_rew_l = [], [], []
        metrics_l: list[dict] = []

        for _ in range(batch_size):
            # ── 1. Host observes and acts ─────────────────────────────────
            h_obs = env.observe("host")
            h_door, _ = self.host_agent.act(h_obs)
            h_frac = h_door / norm

            # ── 2. Expose signal so players condition on it ───────────────
            #    env.step_all() will derive the same value from h_frac,
            #    so this assignment is just an early preview for player obs.
            env._public_signal = h_door

            # ── 3. Players observe (with signal) and act ──────────────────
            p_fracs: dict[str, float] = {}
            for pid in range(N):
                name = f"player_{pid}"
                p_obs = env.observe(name)
                p_door, _ = self.player_agent.act(p_obs)
                p_fracs[name] = p_door / norm
                player_obs_l.append(p_obs)
                player_act_l.append(p_door)

            # ── 4. Step (re-sets signal to same h_door, harmless) ─────────
            rewards = env.step_all(h_frac, p_fracs)

            # ── 5. Record experience ──────────────────────────────────────
            host_obs_l.append(h_obs)
            host_act_l.append(h_door)
            host_rew_l.append(rewards["host"])

            for pid in range(N):
                player_rew_l.append(rewards[f"player_{pid}"])

            metrics_l.append(dict(env.last_round_info))

        return {
            "host": {
                "obs":    np.array(host_obs_l,   dtype=np.float32),
                "action": np.array(host_act_l,   dtype=np.int64),
                "reward": np.array(host_rew_l,   dtype=np.float32),
            },
            "player": {
                "obs":    np.array(player_obs_l,   dtype=np.float32),
                "action": np.array(player_act_l,   dtype=np.int64),
                "reward": np.array(player_rew_l,   dtype=np.float32),
            },
            "metrics": metrics_l,
        }

    # ------------------------------------------------------------------
    # REINFORCE update
    # ------------------------------------------------------------------

    def _reinforce_loss(
        self,
        agent: MlpAgent,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_t: torch.Tensor,
        baseline: float,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Compute REINFORCE loss and update the EMA baseline.

        loss = -mean[(r - b) · log π(a|o)] - β · mean[H[π]]
        Returns (loss, new_baseline, mean_entropy).
        """
        new_baseline = (
            self.baseline_momentum * baseline
            + (1 - self.baseline_momentum) * rew_t.mean().item()
        )
        advantage = rew_t - new_baseline

        log_probs, entropy = agent.evaluate(obs_t, act_t)
        loss = -(advantage * log_probs).mean() - self.entropy_coeff * entropy.mean()
        return loss, new_baseline, float(entropy.mean().item())

    def _update(self, batch: dict) -> dict:
        # ── Host ─────────────────────────────────────────────────────────
        h = batch["host"]
        h_obs_t = torch.FloatTensor(h["obs"])
        h_act_t = torch.LongTensor(h["action"])
        h_rew_t = torch.FloatTensor(h["reward"])

        h_loss, self.host_baseline, h_ent = self._reinforce_loss(
            self.host_agent, h_obs_t, h_act_t, h_rew_t, self.host_baseline
        )
        self.host_opt.zero_grad()
        h_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.host_agent.parameters(), self.grad_clip)
        self.host_opt.step()

        # ── Players (shared network — pool all N·B samples) ───────────────
        p = batch["player"]
        p_obs_t = torch.FloatTensor(p["obs"])
        p_act_t = torch.LongTensor(p["action"])
        p_rew_t = torch.FloatTensor(p["reward"])

        p_loss, self.player_baseline, p_ent = self._reinforce_loss(
            self.player_agent, p_obs_t, p_act_t, p_rew_t, self.player_baseline
        )
        self.player_opt.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.player_agent.parameters(), self.grad_clip)
        self.player_opt.step()

        return {
            "host_loss":     float(h_loss.item()),
            "player_loss":   float(p_loss.item()),
            "host_entropy":  h_ent,
            "player_entropy": p_ent,
        }

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_metrics(metrics_list: list[dict]) -> dict:
        """Compute mean per-round statistics from a batch."""
        win_ratios, host_rews, player_rews = [], [], []
        honesties, follow_rates = [], []

        for m in metrics_list:
            win_ratios.append(m["win_ratio"])
            host_rews.append(m["rewards"]["host"])
            p_rews = [v for k, v in m["rewards"].items() if k != "host"]
            player_rews.append(float(np.mean(p_rews)))

            # Signal honesty: did host point at the correct door?
            honesties.append(int(m["correct_door"] == m["public_signal"]))

            # Follow rate: fraction of players who chose the signaled door
            n = len(m["door_choices"])
            if n > 0:
                followed = sum(
                    1 for door in m["door_choices"].values()
                    if door == m["public_signal"]
                )
                follow_rates.append(followed / n)

        return {
            "win_ratio":      float(np.mean(win_ratios)),
            "host_reward":    float(np.mean(host_rews)),
            "player_reward":  float(np.mean(player_rews)),
            "signal_honesty": float(np.mean(honesties)),
            "follow_rate":    float(np.mean(follow_rates)) if follow_rates else 0.0,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_rounds: int = 100_000,
        batch_size: int = 128,
        log_interval: int = 2_000,
        save_interval: int = 20_000,
        spotlight_interval: int = 10_000,
        save_dir: str = "checkpoints/mlp_reinforce",
    ) -> list[dict]:
        """
        Run REINFORCE training.

        Saves a CSV log to  <save_dir>/training_log.csv  every log_interval.
        Prints a round-level spotlight every spotlight_interval rounds.

        Returns
        -------
        log : list[dict]  — one entry per log_interval rounds
        """
        import csv as _csv

        _CSV_FIELDS = [
            "round", "host_reward", "player_reward", "win_ratio",
            "signal_honesty", "follow_rate",
            "host_loss", "player_loss", "host_entropy", "player_entropy",
            "host_baseline", "player_baseline", "elapsed_s",
        ]

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "training_log.csv")
        self.env.reset()

        log: list[dict] = []
        rounds_done = 0
        t0 = time.time()
        recent_metrics: list[dict] = []  # ring-buffer for spotlight

        with open(csv_path, "w", newline="") as csv_f:
            writer = _csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS,
                                     extrasaction="ignore")
            writer.writeheader()

            while rounds_done < total_rounds:
                batch   = self._collect_batch(batch_size)
                losses  = self._update(batch)
                metrics = self._agg_metrics(batch["metrics"])
                rounds_done += batch_size

                # Keep up to 20 recent rounds for spotlight
                recent_metrics.extend(batch["metrics"])
                if len(recent_metrics) > 20:
                    recent_metrics = recent_metrics[-20:]

                if rounds_done % log_interval < batch_size:
                    elapsed = time.time() - t0
                    entry = {
                        "round":           rounds_done,
                        **metrics,
                        **losses,
                        "host_baseline":   self.host_baseline,
                        "player_baseline": self.player_baseline,
                        "elapsed_s":       elapsed,
                    }
                    log.append(entry)
                    writer.writerow(entry)
                    csv_f.flush()
                    _print_log(entry)

                if spotlight_interval > 0 and rounds_done % spotlight_interval < batch_size:
                    self._print_spotlight(recent_metrics)

                if rounds_done % save_interval < batch_size:
                    _save_checkpoints(self, save_dir, rounds_done)

        print(f"\nDone. {rounds_done} rounds in {time.time()-t0:.1f}s")
        print(f"Log  → {csv_path}")
        return log

    @staticmethod
    def _print_spotlight(metrics_list: list[dict], n: int = 5) -> None:
        """
        Print the last n rounds as a table for qualitative inspection.

        Choices column markers:
          [d]* = player chose the correct door  (green)
          [d]~ = player followed signal (wrong) (yellow)
           d   = player picked some other door
        """
        C, G, R, Y, RST = "\033[96m", "\033[92m", "\033[91m", "\033[93m", "\033[0m"
        recent = metrics_list[-n:]
        if not recent:
            return

        def _mark(c: int, cd: int, sig: int) -> str:
            if c == cd:
                return f"{G}{c}*{RST}"
            if c == sig:
                return f"{Y}{c}~{RST}"
            return f"{c} "

        print(f"\n{C}{'━'*24} Spotlight (last {len(recent)} rounds) {'━'*6}{RST}")
        print(f"  {'Round':>7}  {'Corr':^4}  {'Sig':^3}  {'Hon':^3}  "
              f"{'Choices  (*=correct ~=signal)':<32}  {'Fol':>4}  {'x':>5}  {'H-Rwd':>7}")
        print(f"  {'─'*78}")
        for m in recent:
            cd      = m["correct_door"]
            sig     = m["public_signal"]
            honest  = (cd == sig)
            choices = list(m["door_choices"].values())
            n_fol   = sum(1 for c in choices if c == sig)
            fol_r   = n_fol / len(choices) if choices else 0.0
            x       = m["win_ratio"]
            hr      = m["rewards"]["host"]
            hc      = G if hr >= 0 else R
            hon_s   = f"{G}Y{RST}" if honest else f"{R}N{RST}"
            ch_s    = " ".join(_mark(c, cd, sig) for c in choices)
            print(
                f"  {m['round']:>7}  {cd:^4}  {Y}{sig:^3}{RST}  {hon_s}   "
                f"{ch_s}   {fol_r:>4.2f}  {x:>5.3f}  {hc}{hr:>+7.2f}{RST}"
            )
        print(f"{C}{'━'*80}{RST}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_log(e: dict) -> None:
    G, R, D, RST = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
    hr = e["host_reward"]
    pr = e["player_reward"]
    hc = G if hr >= 0 else R
    pc = G if pr >= 0 else R
    h_ent  = e.get("host_entropy",   float("nan"))
    p_ent  = e.get("player_entropy", float("nan"))
    h_loss = e.get("host_loss",      float("nan"))
    p_loss = e.get("player_loss",    float("nan"))
    print(
        f"[{e['round']:>8}]  "
        f"H={hc}{hr:+.3f}{RST}  P={pc}{pr:+.3f}{RST}  "
        f"wr={e['win_ratio']:.3f}  "
        f"hon={e['signal_honesty']:.2f}  fol={e['follow_rate']:.2f}  "
        f"{D}loss=({h_loss:.3f},{p_loss:.3f})  "
        f"ent=({h_ent:.3f},{p_ent:.3f}){RST}  "
        f"({e['elapsed_s']:.0f}s)"
    )


def _save_checkpoints(runner: ReinforceRunner, save_dir: str, rounds: int) -> None:
    torch.save(
        runner.host_agent.state_dict(),
        os.path.join(save_dir, f"host_{rounds}.pt"),
    )
    torch.save(
        runner.player_agent.state_dict(),
        os.path.join(save_dir, f"player_{rounds}.pt"),
    )
    print(f"  [ckpt] saved at round {rounds}")
