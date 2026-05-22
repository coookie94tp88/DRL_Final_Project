"""
CurriculumRunner: Three-Phase Curriculum Training for OracleGambit
===================================================================

Storyline (from the design document):

  Phase A — Warm-Up
    Host is forced to be honest (always emits correct_door as signal).
    Players receive a reliable signal and learn to follow it.
    Stop early when: follow_rate >= target_follow_rate_a

  Phase B — Adversarial
    Players are frozen with their "trusting" policy from Phase A.
    Host trains freely and learns to exploit Player trust.
    Stop early when: signal_honesty <= target_honesty_b

  Phase C — Joint (Arms-Race)
    Both agents train simultaneously via REINFORCE.
    Host can't always deceive → Players adapt → signal honesty oscillates.

Expected metric trajectory:
  A: fol 0.25→0.65+,  wr  0.25→0.65+   (Players start trusting)
  B: hon 1.0→0.35-,   wr  0.65→~0.25   (Host learns to deceive)
  C: hon oscillates,   fol oscillates    (Arms-race; no stable equilibrium)

Key implementation notes
------------------------
* Phase A: host_agent is NOT called; h_door = env._correct_door
* Phase B: player_agent.act() IS called (observes signal) but no grad update
* Phase C: identical to standard ReinforceRunner
* Stop conditions are evaluated at log_interval boundaries on a fresh window
  of metrics (not on single-batch noise).
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable

import numpy as np
import torch
import torch.optim as optim

from agents.mlp_agent import MlpAgent


_CSV_FIELDS = [
    "phase", "round", "host_reward", "player_reward", "win_ratio",
    "signal_honesty", "follow_rate",
    "host_loss", "player_loss", "host_entropy", "player_entropy",
    "host_baseline", "player_baseline", "elapsed_s",
]

_PHASE_COLOR = {
    "A": "\033[94m",   # blue
    "B": "\033[91m",   # red
    "C": "\033[93m",   # yellow/orange
}
RST = "\033[0m"
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
DIM = "\033[2m"


class CurriculumRunner:
    """
    Three-phase curriculum trainer for OracleGambit.

    Phase A : Honest Host  → Players learn to follow the signal.
    Phase B : Frozen Players → Host learns to exploit their trust.
    Phase C : Joint REINFORCE → Arms-race dynamics emerge.
    """

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
        self.env              = env
        self.host_agent       = host_agent
        self.player_agent     = player_agent
        self.entropy_coeff    = entropy_coeff
        self.baseline_momentum = baseline_momentum
        self.grad_clip        = grad_clip

        self.host_opt   = optim.Adam(host_agent.parameters(),   lr=lr_host)
        self.player_opt = optim.Adam(player_agent.parameters(), lr=lr_player)

        self.host_baseline:   float = 0.0
        self.player_baseline: float = 0.0

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def _collect_batch(self, batch_size: int, honest_host: bool = False) -> dict:
        """
        Collect `batch_size` rounds of experience.

        Parameters
        ----------
        honest_host : bool
            Phase A only.  If True, bypass the host policy and always emit
            env._correct_door as the public signal.  The host network is NOT
            called, so no host gradients are produced.
        """
        env  = self.env
        N    = env.num_players
        D    = env.num_doors
        norm = D - 1 if D > 1 else 1

        host_obs_l, host_act_l, host_rew_l = [], [], []
        player_obs_l, player_act_l, player_rew_l = [], [], []
        metrics_l: list[dict] = []

        for _ in range(batch_size):
            # ── 1. Host phase ─────────────────────────────────────────────
            h_obs  = env.observe("host")
            if honest_host:
                h_door = env._correct_door          # bypass policy
            else:
                h_door, _ = self.host_agent.act(h_obs)
            h_frac = h_door / norm
            env._public_signal = h_door             # expose before players observe

            # ── 2. Players observe (with current signal) and act ──────────
            p_fracs: dict[str, float] = {}
            for pid in range(N):
                name  = f"player_{pid}"
                p_obs = env.observe(name)
                p_door, _ = self.player_agent.act(p_obs)
                p_fracs[name] = p_door / norm
                player_obs_l.append(p_obs)
                player_act_l.append(p_door)

            # ── 3. Step ───────────────────────────────────────────────────
            rewards = env.step_all(h_frac, p_fracs)

            # ── 4. Record ─────────────────────────────────────────────────
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
                "obs":    np.array(player_obs_l,  dtype=np.float32),
                "action": np.array(player_act_l,  dtype=np.int64),
                "reward": np.array(player_rew_l,  dtype=np.float32),
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
        """REINFORCE loss with EMA baseline and entropy bonus."""
        new_baseline = (
            self.baseline_momentum * baseline
            + (1 - self.baseline_momentum) * rew_t.mean().item()
        )
        advantage = rew_t - new_baseline
        log_probs, entropy = agent.evaluate(obs_t, act_t)
        loss = -(advantage * log_probs).mean() - self.entropy_coeff * entropy.mean()
        return loss, new_baseline, float(entropy.mean().item())

    def _update(
        self,
        batch: dict,
        train_host: bool = True,
        train_players: bool = True,
    ) -> dict:
        """
        Gradient update step.

        Parameters
        ----------
        train_host    : False in Phase A  (players only)
        train_players : False in Phase B  (host only)
        """
        result: dict = {
            "host_loss":      float("nan"),
            "player_loss":    float("nan"),
            "host_entropy":   float("nan"),
            "player_entropy": float("nan"),
        }

        if train_host:
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
            result["host_loss"]    = float(h_loss.item())
            result["host_entropy"] = h_ent

        if train_players:
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
            result["player_loss"]    = float(p_loss.item())
            result["player_entropy"] = p_ent

        return result

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_metrics(metrics_list: list[dict]) -> dict:
        """Aggregate per-round info dicts into scalar statistics."""
        win_ratios, host_rews, player_rews = [], [], []
        honesties, follow_rates = [], []

        for m in metrics_list:
            win_ratios.append(m["win_ratio"])
            host_rews.append(m["rewards"]["host"])
            p_rews = [v for k, v in m["rewards"].items() if k != "host"]
            player_rews.append(float(np.mean(p_rews)))
            honesties.append(int(m["correct_door"] == m["public_signal"]))
            n = len(m["door_choices"])
            if n > 0:
                followed = sum(
                    1 for d in m["door_choices"].values() if d == m["public_signal"]
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
    # Generic phase runner
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase: str,
        max_rounds: int,
        batch_size: int,
        honest_host: bool,
        train_host: bool,
        train_players: bool,
        stop_fn: Callable[[dict, dict], bool] | None,
        log_interval: int,
        spotlight_interval: int,
        save_dir: str,
        writer,
        rounds_offset: int,
        t0: float,
        min_rounds_before_stop: int = 0,
    ) -> tuple[int, list[dict]]:
        """
        Run one curriculum phase.

        Stop conditions are evaluated at log_interval boundaries using a fresh
        window of metrics (avoids triggering on single-batch noise).

        stop_fn signature: (agg_metrics: dict, last_losses: dict) -> bool
          agg_metrics  — averaged over the current log window
          last_losses  — from the most recent gradient update (has host_entropy etc.)

        min_rounds_before_stop : int
          Phase will not early-stop before this many rounds have been completed,
          even if stop_fn returns True.  Use for Phase B to ensure the host
          actually *learns* deception rather than triggering on random-init noise.

        Returns
        -------
        (rounds_completed_this_phase, log_entries)
        """
        _print_phase_header(phase)

        rounds_done    = 0
        log_entries: list[dict] = []
        window_metrics: list[dict] = []   # reset at each log boundary
        recent_metrics: list[dict] = []   # last ≤20 rounds for spotlight

        while rounds_done < max_rounds:
            batch  = self._collect_batch(batch_size, honest_host=honest_host)
            losses = self._update(batch, train_host=train_host, train_players=train_players)
            rounds_done += batch_size

            window_metrics.extend(batch["metrics"])
            recent_metrics.extend(batch["metrics"])
            if len(recent_metrics) > 20:
                recent_metrics = recent_metrics[-20:]

            global_round = rounds_offset + rounds_done

            # ── Log at interval boundary ──────────────────────────────────
            if rounds_done % log_interval < batch_size:
                agg     = self._agg_metrics(window_metrics)
                elapsed = time.time() - t0

                entry = {
                    "phase":           phase,
                    "round":           global_round,
                    **agg,
                    **losses,
                    "host_baseline":   self.host_baseline,
                    "player_baseline": self.player_baseline,
                    "elapsed_s":       elapsed,
                }
                log_entries.append(entry)
                writer.writerow(entry)
                _print_log(phase, entry)

                window_metrics.clear()

                # Evaluate stop condition on the just-logged window.
                # Guard: must have completed min_rounds_before_stop first.
                if (stop_fn is not None
                        and rounds_done >= min_rounds_before_stop
                        and stop_fn(agg, losses)):
                    pc = _PHASE_COLOR.get(phase, RST)
                    print(f"\n  {pc}[Phase {phase}] Early stop at {rounds_done:,} rounds.{RST}")
                    break

            # ── Spotlight ─────────────────────────────────────────────────
            if spotlight_interval > 0 and rounds_done % spotlight_interval < batch_size:
                _print_spotlight(recent_metrics, phase=phase)

            # ── Checkpoint ────────────────────────────────────────────────
            if save_dir and rounds_done % 20_000 < batch_size:
                _save_checkpoints(self, save_dir, global_round, phase)

        _print_phase_summary(phase, log_entries)
        return rounds_done, log_entries

    # ------------------------------------------------------------------
    # Full curriculum
    # ------------------------------------------------------------------

    def run_curriculum(
        self,
        rounds_a: int = 50_000,
        rounds_b: int = 50_000,
        rounds_c: int = 100_000,
        batch_size: int = 128,
        log_interval: int = 2_000,
        spotlight_interval: int = 10_000,
        target_follow_rate_a: float = 0.65,
        target_honesty_b: float = 0.35,
        target_entropy_b: float = 0.9,
        min_rounds_b: int = 20_000,
        save_dir: str = "checkpoints/curriculum",
    ) -> list[dict]:
        """
        Run the full 3-phase curriculum training.

        Phase A — honest host, train players only
            Stops early when: follow_rate >= target_follow_rate_a
        Phase B — frozen players, train host only
            Stops early when: signal_honesty <= target_honesty_b
                          AND host_entropy   <= target_entropy_b   (policy concentrated)
                          AND rounds_done    >= min_rounds_b        (actually learned)
        Phase C — joint training, no early stop

        target_entropy_b : float
            Host entropy must drop BELOW this value before Phase B can stop.
            Default 0.9 < ln(4)≈1.386 ensures the host has actually converged
            to a deceptive strategy (not just random-init noise).
        min_rounds_b : int
            Phase B will not stop before this many rounds even if both metric
            thresholds are already satisfied.  Prevents Phase B from exiting
            immediately when the untrained host accidentally has low honesty.

        Returns
        -------
        total_log : list[dict]  — one entry per log_interval rounds (all phases)
        """
        import csv as _csv

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "curriculum_log.csv")

        self.env.reset()
        t0         = time.time()
        total_log: list[dict] = []

        with open(csv_path, "w", newline="") as csv_f:
            writer = _csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()

            # ── Phase A: Warm-Up ──────────────────────────────────────────
            r_a, log_a = self._run_phase(
                "A", rounds_a, batch_size,
                honest_host=True, train_host=False, train_players=True,
                stop_fn=lambda m, l: m["follow_rate"] >= target_follow_rate_a,
                log_interval=log_interval,
                spotlight_interval=spotlight_interval,
                save_dir=save_dir, writer=writer,
                rounds_offset=0, t0=t0,
            )
            total_log.extend(log_a)

            # ── Phase B: Adversarial ──────────────────────────────────────
            # Stop only when BOTH conditions hold (AND min rounds elapsed):
            #   hon  <= target_honesty_b  — host is sending mostly wrong signals
            #   ent  <= target_entropy_b  — host policy has actually concentrated
            #                               (not just random-init noise)
            r_b, log_b = self._run_phase(
                "B", rounds_b, batch_size,
                honest_host=False, train_host=True, train_players=False,
                stop_fn=lambda m, l: (
                    m["signal_honesty"] <= target_honesty_b
                    and not math.isnan(l.get("host_entropy", float("nan")))
                    and l["host_entropy"] <= target_entropy_b
                ),
                log_interval=log_interval,
                spotlight_interval=spotlight_interval,
                save_dir=save_dir, writer=writer,
                rounds_offset=r_a, t0=t0,
                min_rounds_before_stop=min_rounds_b,
            )
            total_log.extend(log_b)

            # ── Phase C: Joint (Arms-Race) ────────────────────────────────
            r_c, log_c = self._run_phase(
                "C", rounds_c, batch_size,
                honest_host=False, train_host=True, train_players=True,
                stop_fn=None,
                log_interval=log_interval,
                spotlight_interval=spotlight_interval,
                save_dir=save_dir, writer=writer,
                rounds_offset=r_a + r_b, t0=t0,
            )
            total_log.extend(log_c)

        total = r_a + r_b + r_c
        print(f"\nCurriculum complete. {total:,} rounds in {time.time()-t0:.1f}s")
        print(f"Log  → {csv_path}")
        return total_log


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_phase_header(phase: str) -> None:
    pc = _PHASE_COLOR.get(phase, RST)
    labels = {
        "A": "Warm-Up      | Honest Host → Players learn to follow signal",
        "B": "Adversarial  | Players frozen → Host learns to deceive",
        "C": "Joint        | Arms-race dynamics (both agents train)",
    }
    print(f"\n{pc}{'━'*64}{RST}")
    print(f"{pc}  Phase {phase}: {labels.get(phase, '')}{RST}")
    print(f"{pc}{'━'*64}{RST}")


def _print_phase_summary(phase: str, log_entries: list[dict]) -> None:
    if not log_entries:
        return
    pc   = _PHASE_COLOR.get(phase, RST)
    last = log_entries[-1]
    print(
        f"\n{pc}  Phase {phase} summary — "
        f"wr={last['win_ratio']:.3f}  "
        f"hon={last['signal_honesty']:.2f}  "
        f"fol={last['follow_rate']:.2f}  "
        f"({last['elapsed_s']:.0f}s elapsed){RST}\n"
    )


def _print_log(phase: str, e: dict) -> None:
    pc     = _PHASE_COLOR.get(phase, RST)
    hr     = e["host_reward"]
    pr     = e["player_reward"]
    hc     = G if hr >= 0 else R
    pc2    = G if pr >= 0 else R
    h_ent  = e.get("host_entropy",   float("nan"))
    p_ent  = e.get("player_entropy", float("nan"))
    h_loss = e.get("host_loss",      float("nan"))
    p_loss = e.get("player_loss",    float("nan"))
    t_s    = e.get("elapsed_s",      0.0)
    print(
        f"  {pc}[{phase}]{RST} "
        f"[{e['round']:>7}]  "
        f"H={hc}{hr:>+6.3f}{RST}  "
        f"P={pc2}{pr:>+6.3f}{RST}  "
        f"wr={e['win_ratio']:.3f}  "
        f"hon={e['signal_honesty']:.2f}  "
        f"fol={e['follow_rate']:.2f}  "
        f"{DIM}loss=({h_loss:.3f},{p_loss:.3f})  "
        f"ent=({h_ent:.3f},{p_ent:.3f})  "
        f"({t_s:.0f}s){RST}"
    )


def _print_spotlight(metrics_list: list[dict], n: int = 5, phase: str = "") -> None:
    C      = _PHASE_COLOR.get(phase, "\033[96m")
    recent = metrics_list[-n:]
    if not recent:
        return

    def _mark(c: int, cd: int, sig: int) -> str:
        if c == cd:
            return f"{G}{c}*{RST}"
        if c == sig:
            return f"{Y}{c}~{RST}"
        return f"{c} "

    tag = f" Phase {phase}" if phase else ""
    print(f"\n{C}{'━'*24} Spotlight{tag} (last {len(recent)} rounds)  *=correct  ~=signal {'━'*4}{RST}")
    print(
        f"  {'Round':>7}  {'Corr':^4}  {'Sig':^3}  {'Hon':^3}  "
        f"{'Choices  (*=correct ~=signal)':<32}  {'Fol':>4}  {'x':>5}  {'H-Rwd':>7}"
    )
    print(f"  {'─'*78}")
    for m in recent:
        cd     = m["correct_door"]
        sig    = m["public_signal"]
        honest = (cd == sig)
        choices = list(m["door_choices"].values())
        n_fol  = sum(1 for c in choices if c == sig)
        fol_r  = n_fol / len(choices) if choices else 0.0
        x      = m["win_ratio"]
        hr     = m["rewards"]["host"]
        hc     = G if hr >= 0 else R
        hon_s  = f"{G}Y{RST}" if honest else f"{R}N{RST}"
        ch_s   = " ".join(_mark(c, cd, sig) for c in choices)
        print(
            f"  {m['round']:>7}  {cd:^4}  {Y}{sig:^3}{RST}  {hon_s}   "
            f"{ch_s}   {fol_r:>4.2f}  {x:>5.3f}  {hc}{hr:>+7.2f}{RST}"
        )
    print(f"{C}{'━'*80}{RST}\n")


def _save_checkpoints(runner: CurriculumRunner, save_dir: str, rounds_done: int, phase: str) -> None:
    tag = f"phase{phase}_{rounds_done:07d}"
    torch.save(runner.host_agent.state_dict(),
               os.path.join(save_dir, f"host_{tag}.pt"))
    torch.save(runner.player_agent.state_dict(),
               os.path.join(save_dir, f"player_{tag}.pt"))
    print(f"  Saved checkpoints: {tag}")
