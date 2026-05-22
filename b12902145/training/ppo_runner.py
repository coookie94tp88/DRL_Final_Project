"""
PPO (Proximal Policy Optimization) runner for OracleGambit with TransformerAgent.

Algorithm overview
------------------
For each batch of B rounds:
  1. Two-phase collection:
       a. Host observes → act() → (h_door, old_log_prob, value)
       b. env._public_signal = h_door  (players see signal)
       c. Each player observes → act() → (p_door, old_log_prob, value)
       d. env.step_all() → rewards
  2. After B rounds, get bootstrap values V(s_{B+1}) for GAE.
  3. GAE advantage estimation (γ, λ) for host and per-player trajectories.
  4. K PPO epochs of clipped surrogate + value + entropy loss.

Key design decisions
--------------------
* Parameter sharing — all N players share one TransformerAgent.
* GAE over the B-round mini-batch trajectory:
    - Host: 1-D sequence of B rewards / values.
    - Players: (B × N) → compute GAE per player (column-wise), then flatten.
* Advantage normalisation before each PPO update.
* Gradient clipping (L2 ≤ 1.0) for training stability.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.transformer_agent import TransformerAgent


class PPORunner:
    def __init__(
        self,
        env,
        host_agent: TransformerAgent,
        player_agent: TransformerAgent,
        lr_host: float = 3e-4,
        lr_player: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
        grad_clip: float = 1.0,
        minibatch_size: int = 256,   # mini-batch size for PPO update; improves CPU cache efficiency
    ) -> None:
        self.env = env
        self.host_agent   = host_agent
        self.player_agent = player_agent
        self.clip_eps      = clip_eps
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.value_coeff   = value_coeff
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs    = ppo_epochs
        self.grad_clip     = grad_clip
        self.minibatch_size = minibatch_size

        self.host_opt   = optim.Adam(host_agent.parameters(),   lr=lr_host)
        self.player_opt = optim.Adam(player_agent.parameters(), lr=lr_player)

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def _collect_batch(self, batch_size: int, honest_host: bool = False) -> dict:
        """
        Run `batch_size` rounds and record experience for PPO.
        If honest_host=True, the host always signals the correct door (Phase A).

        Returns
        -------
        batch : {
          "host"   : obs(B,D), action(B,), reward(B,), old_log_prob(B,), value(B,)
          "player" : obs(B*N,D), action(B*N,), reward(B*N,), old_log_prob(B*N,),
                     value(B*N,)    [interleaved: round0_p0…p5, round1_p0…p5, …]
          "player_rewards_2d" : (B, N)  — per-player per-round rewards for GAE
          "player_values_2d"  : (B, N)  — per-player per-round V(s) for GAE
          "metrics" : list[dict]
          "host_boot_obs"   : np.ndarray  — host obs AFTER last round (bootstrap)
          "player_boot_obs" : np.ndarray  — player_0 obs AFTER last round (bootstrap)
        }
        """
        env = self.env
        N = env.num_players
        D = env.num_doors
        norm = D - 1 if D > 1 else 1

        host_obs_l, host_act_l, host_rew_l, host_lp_l, host_val_l = [], [], [], [], []
        p_obs_l, p_act_l, p_rew_l, p_lp_l, p_val_l = [], [], [], [], []
        # Per-round arrays for GAE (shape: B×N after collection)
        p_rew2d, p_val2d = [], []
        metrics_l: list[dict] = []

        for _ in range(batch_size):
            # ── Host: observe → act ───────────────────────────────────────
            h_obs = env.observe("host")
            if honest_host:
                h_door = env._correct_door      # Phase A: bypass policy, signal truth
                h_lp, h_val = 0.0, 0.0         # placeholders (train_host=False in Phase A)
            else:
                h_door, h_lp, h_val = self.host_agent.act(h_obs)
            h_frac = h_door / norm

            env._public_signal = h_door   # expose signal so players can condition on it

            # ── Players: batch all N obs into ONE forward pass ────────────
            # Collecting obs individually (env has per-player private state),
            # but running inference as a single (N, obs_dim) batch is ~3-4x
            # faster than N separate act() calls on CPU.
            p_obs_batch = np.stack(
                [env.observe(f"player_{pid}") for pid in range(N)]
            )  # (N, obs_dim)
            with torch.no_grad():
                _p_obs_t = torch.FloatTensor(p_obs_batch)
                _p_dist, _p_vals_t = self.player_agent(_p_obs_t)  # (N,D), (N,)
                _p_acts_t = _p_dist.sample()                       # (N,)
                _p_lps_t  = _p_dist.log_prob(_p_acts_t)           # (N,)
            _p_doors = _p_acts_t.numpy()
            _p_lps   = _p_lps_t.numpy()
            _p_vals  = _p_vals_t.numpy()

            p_fracs: dict[str, float] = {
                f"player_{pid}": int(_p_doors[pid]) / norm for pid in range(N)
            }
            round_p_rews: list[float] = []
            round_p_vals: list[float] = []
            for pid in range(N):
                p_obs_l.append(p_obs_batch[pid])
                p_act_l.append(int(_p_doors[pid]))
                p_lp_l.append(float(_p_lps[pid]))
                p_val_l.append(float(_p_vals[pid]))
                round_p_vals.append(float(_p_vals[pid]))

            # ── Step ──────────────────────────────────────────────────────
            rewards = env.step_all(h_frac, p_fracs)

            # ── Record ────────────────────────────────────────────────────
            host_obs_l.append(h_obs)
            host_act_l.append(h_door)
            host_rew_l.append(rewards["host"])
            host_lp_l.append(h_lp)
            host_val_l.append(h_val)

            for pid in range(N):
                r = rewards[f"player_{pid}"]
                p_rew_l.append(r)
                round_p_rews.append(r)

            p_rew2d.append(round_p_rews)
            p_val2d.append(round_p_vals)
            metrics_l.append(dict(env.last_round_info))

        # Bootstrap observations (state AFTER the last collected round)
        host_boot_obs   = env.observe("host")
        player_boot_obs = env.observe("player_0")

        return {
            "host": {
                "obs":          np.array(host_obs_l,  dtype=np.float32),
                "action":       np.array(host_act_l,  dtype=np.int64),
                "reward":       np.array(host_rew_l,  dtype=np.float32),
                "old_log_prob": np.array(host_lp_l,   dtype=np.float32),
                "value":        np.array(host_val_l,  dtype=np.float32),
            },
            "player": {
                "obs":          np.array(p_obs_l,  dtype=np.float32),
                "action":       np.array(p_act_l,  dtype=np.int64),
                "reward":       np.array(p_rew_l,  dtype=np.float32),
                "old_log_prob": np.array(p_lp_l,   dtype=np.float32),
                "value":        np.array(p_val_l,  dtype=np.float32),
            },
            "player_rewards_2d": np.array(p_rew2d,  dtype=np.float32),  # (B, N)
            "player_values_2d":  np.array(p_val2d,  dtype=np.float32),  # (B, N)
            "host_boot_obs":   host_boot_obs,
            "player_boot_obs": player_boot_obs,
            "metrics": metrics_l,
        }

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: np.ndarray,   # (B,)
        values:  np.ndarray,   # (B,)
        bootstrap_val: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalised Advantage Estimation over a B-step trajectory.

        Returns
        -------
        advantages : (B,)  — A_t = sum_l (γλ)^l δ_{t+l}
        returns    : (B,)  — discounted return targets for the value head
        """
        B   = len(rewards)
        adv = np.zeros(B, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(B)):
            next_val = bootstrap_val if t == B - 1 else float(values[t + 1])
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae   = delta + self.gamma * self.gae_lambda * gae
            adv[t] = gae
        return adv, adv + values

    # ------------------------------------------------------------------
    # PPO loss & update
    # ------------------------------------------------------------------

    def _ppo_update(
        self,
        agent: TransformerAgent,
        optimizer: optim.Optimizer,
        obs_t:     torch.Tensor,    # (B, obs_dim)
        act_t:     torch.Tensor,    # (B,)
        old_lp_t:  torch.Tensor,    # (B,)
        adv_t:     torch.Tensor,    # (B,)  already normalised
        ret_t:     torch.Tensor,    # (B,)
    ) -> tuple[float, float, float]:
        """
        Run `ppo_epochs` of clipped-surrogate PPO on the given data.

        Returns (mean_actor_loss, mean_value_loss, mean_entropy) over updates.
        """
        B  = obs_t.size(0)
        mb = B if self.minibatch_size <= 0 else min(self.minibatch_size, B)
        actor_loss_sum = value_loss_sum = entropy_sum = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B)
            for start in range(0, B, mb):
                idx = perm[start : start + mb]

                new_lp, entropy, new_vals = agent.evaluate(obs_t[idx], act_t[idx])

                # Importance-sampling ratio
                ratio  = (new_lp - old_lp_t[idx]).exp()

                # Clipped surrogate objective
                surr1  = ratio * adv_t[idx]
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE against GAE returns)
                value_loss = F.mse_loss(new_vals, ret_t[idx])

                loss = (actor_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_clip)
                optimizer.step()

                actor_loss_sum += actor_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum    += entropy.mean().item()
                n_updates += 1

        d = max(1, n_updates)
        return actor_loss_sum / d, value_loss_sum / d, entropy_sum / d

    def _update(
        self,
        batch: dict,
        train_host: bool = True,
        train_players: bool = True,
    ) -> dict:
        """Compute GAE, normalise advantages, run PPO.

        Set train_host=False or train_players=False to freeze that agent
        (used during curriculum phases A and B respectively).
        """
        out: dict = {
            "host_loss": float("nan"), "player_loss": float("nan"),
            "host_entropy": float("nan"), "player_entropy": float("nan"),
            "host_val_loss": float("nan"), "player_val_loss": float("nan"),
        }

        # ── Host ─────────────────────────────────────────────────────────
        if train_host:
            hd = batch["host"]
            h_boot = self.host_agent.value(batch["host_boot_obs"])
            h_adv, h_ret = self._compute_gae(hd["reward"], hd["value"], h_boot)
            h_adv = (h_adv - h_adv.mean()) / (h_adv.std() + 1e-8)

            h_obs_t    = torch.FloatTensor(hd["obs"])
            h_act_t    = torch.LongTensor(hd["action"])
            h_old_lp_t = torch.FloatTensor(hd["old_log_prob"])
            h_adv_t    = torch.FloatTensor(h_adv)
            h_ret_t    = torch.FloatTensor(h_ret)

            h_actor_loss, h_val_loss, h_ent = self._ppo_update(
                self.host_agent, self.host_opt,
                h_obs_t, h_act_t, h_old_lp_t, h_adv_t, h_ret_t,
            )
            out["host_loss"]     = h_actor_loss + self.value_coeff * h_val_loss
            out["host_entropy"]  = h_ent
            out["host_val_loss"] = h_val_loss

        # ── Players (parameter sharing: pool all B×N samples) ─────────────
        if train_players:
            p_rew2d = batch["player_rewards_2d"]   # (B, N)
            p_val2d = batch["player_values_2d"]    # (B, N)
            p_boot  = self.player_agent.value(batch["player_boot_obs"])

            B, Np = p_rew2d.shape
            p_adv2d = np.zeros_like(p_rew2d)
            p_ret2d = np.zeros_like(p_rew2d)
            for pid in range(Np):
                p_adv2d[:, pid], p_ret2d[:, pid] = self._compute_gae(
                    p_rew2d[:, pid], p_val2d[:, pid], p_boot
                )

            p_adv_flat = p_adv2d.flatten()
            p_ret_flat = p_ret2d.flatten()
            p_adv_flat = (p_adv_flat - p_adv_flat.mean()) / (p_adv_flat.std() + 1e-8)

            pd = batch["player"]
            p_obs_t    = torch.FloatTensor(pd["obs"])
            p_act_t    = torch.LongTensor(pd["action"])
            p_old_lp_t = torch.FloatTensor(pd["old_log_prob"])
            p_adv_t    = torch.FloatTensor(p_adv_flat)
            p_ret_t    = torch.FloatTensor(p_ret_flat)

            p_actor_loss, p_val_loss, p_ent = self._ppo_update(
                self.player_agent, self.player_opt,
                p_obs_t, p_act_t, p_old_lp_t, p_adv_t, p_ret_t,
            )
            out["player_loss"]     = p_actor_loss + self.value_coeff * p_val_loss
            out["player_entropy"]  = p_ent
            out["player_val_loss"] = p_val_loss

        return out

    # ------------------------------------------------------------------
    # Metrics aggregation (identical to ReinforceRunner)
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_metrics(metrics_list: list[dict]) -> dict:
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
    # Spotlight (same as ReinforceRunner)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_spotlight(metrics_list: list[dict], n: int = 5) -> None:
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
        save_dir: str = "checkpoints/transformer_ppo",
    ) -> list[dict]:
        """
        Run PPO training for `total_rounds` rounds.

        Interface identical to ReinforceRunner.train() for easy comparison.

        Saves
        -----
        * <save_dir>/training_log.csv  — one row per log_interval
        * <save_dir>/host_<round>.pt / player_<round>.pt  — checkpoints
        """
        import csv as _csv

        _CSV_FIELDS = [
            "round", "host_reward", "player_reward", "win_ratio",
            "signal_honesty", "follow_rate",
            "host_loss", "player_loss",
            "host_entropy", "player_entropy",
            "host_val_loss", "player_val_loss",
            "elapsed_s",
        ]

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "training_log.csv")
        self.env.reset()

        log: list[dict] = []
        rounds_done = 0
        t0 = time.time()
        recent_metrics: list[dict] = []

        with open(csv_path, "w", newline="") as csv_f:
            writer = _csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS,
                                     extrasaction="ignore")
            writer.writeheader()

            while rounds_done < total_rounds:
                batch   = self._collect_batch(batch_size)
                losses  = self._update(batch)
                metrics = self._agg_metrics(batch["metrics"])
                rounds_done += batch_size

                recent_metrics.extend(batch["metrics"])
                if len(recent_metrics) > 20:
                    recent_metrics = recent_metrics[-20:]

                if rounds_done % log_interval < batch_size:
                    elapsed = time.time() - t0
                    entry = {
                        "round":   rounds_done,
                        **metrics,
                        **losses,
                        "elapsed_s": elapsed,
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

    # ------------------------------------------------------------------
    # Curriculum phase runner
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_name: str,
        max_rounds: int,
        batch_size: int,
        honest_host: bool,
        train_host: bool,
        train_players: bool,
        stop_fn,                # callable(agg, losses)->bool  or  None
        log_interval: int,
        spotlight_interval: int,
        save_dir: str,
        rounds_offset: int,
        t0: float,
        csv_writer,
        csv_file,
        min_rounds: int = 0,
    ) -> int:
        """Run one curriculum phase.  Returns updated rounds_offset."""
        C, RST = "\033[96m", "\033[0m"
        print(f"\n{C}{'━'*60}{RST}")
        print(f"{C}  Phase {phase_name}  — "
              f"honest_host={honest_host}  train_host={train_host}  "
              f"train_players={train_players}{RST}")
        print(f"{C}{'━'*60}{RST}")

        rounds_done = 0
        recent_metrics: list[dict] = []
        window_metrics: list[dict] = []
        window_losses:  list[dict] = []

        while rounds_done < max_rounds:
            batch   = self._collect_batch(batch_size, honest_host=honest_host)
            losses  = self._update(batch, train_host=train_host,
                                   train_players=train_players)
            metrics = self._agg_metrics(batch["metrics"])
            rounds_done += batch_size

            recent_metrics.extend(batch["metrics"])
            if len(recent_metrics) > 20:
                recent_metrics = recent_metrics[-20:]
            window_metrics.append(metrics)
            window_losses.append(losses)

            total = rounds_offset + rounds_done

            if total % log_interval < batch_size:
                agg = {k: float(np.mean([m[k] for m in window_metrics]))
                       for k in window_metrics[0]}
                agg_l: dict = {}
                for k in window_losses[0]:
                    vals = [l[k] for l in window_losses
                            if not (isinstance(l[k], float) and np.isnan(l[k]))]
                    agg_l[k] = float(np.mean(vals)) if vals else float("nan")
                window_metrics.clear()
                window_losses.clear()

                entry = {
                    "phase": phase_name, "round": total,
                    **agg, **agg_l,
                    "elapsed_s": time.time() - t0,
                }
                csv_writer.writerow(entry)
                csv_file.flush()
                _print_log_curriculum(entry)

                if spotlight_interval > 0 and total % spotlight_interval < batch_size:
                    self._print_spotlight(recent_metrics)

                if stop_fn is not None and rounds_done >= min_rounds:
                    if stop_fn(agg, agg_l):
                        print(f"  Phase {phase_name}: early stop at round {total}")
                        break

            if total % 20_000 < batch_size:
                _save_checkpoints(self, save_dir, total)

        return rounds_offset + rounds_done

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
        save_dir: str = "checkpoints/transformer_ppo_curriculum",
    ) -> None:
        """
        Three-phase curriculum for Transformer+PPO.

        Phase A  Honest host, train players only.
                 Stops when follow_rate >= target_follow_rate_a.
        Phase B  Deceptive host, train host only (players frozen).
                 Stops when hon <= target_honesty_b AND
                             host_entropy <= target_entropy_b AND
                             rounds >= min_rounds_b.
        Phase C  Joint training for rounds_c rounds (no early stop).
        """
        import csv as _csv

        _CSV_FIELDS = [
            "phase", "round", "host_reward", "player_reward", "win_ratio",
            "signal_honesty", "follow_rate",
            "host_loss", "player_loss",
            "host_entropy", "player_entropy",
            "host_val_loss", "player_val_loss",
            "elapsed_s",
        ]

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "training_log.csv")
        self.env.reset()
        t0 = time.time()
        offset = 0

        with open(csv_path, "w", newline="") as csv_f:
            writer = _csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS,
                                     extrasaction="ignore")
            writer.writeheader()

            # ── Phase A ────────────────────────────────────────────────────
            offset = self._run_phase(
                "A", rounds_a, batch_size,
                honest_host=True, train_host=False, train_players=True,
                stop_fn=lambda m, _: m["follow_rate"] >= target_follow_rate_a,
                log_interval=log_interval, spotlight_interval=spotlight_interval,
                save_dir=save_dir, rounds_offset=offset, t0=t0,
                csv_writer=writer, csv_file=csv_f, min_rounds=0,
            )

            # ── Phase B ────────────────────────────────────────────────────
            def _stop_b(m: dict, l: dict) -> bool:
                h_ent = l.get("host_entropy", float("nan"))
                ent_ok = np.isnan(h_ent) or h_ent <= target_entropy_b
                return m["signal_honesty"] <= target_honesty_b and ent_ok

            offset = self._run_phase(
                "B", rounds_b, batch_size,
                honest_host=False, train_host=True, train_players=False,
                stop_fn=_stop_b,
                log_interval=log_interval, spotlight_interval=spotlight_interval,
                save_dir=save_dir, rounds_offset=offset, t0=t0,
                csv_writer=writer, csv_file=csv_f, min_rounds=min_rounds_b,
            )

            # ── Phase C ────────────────────────────────────────────────────
            offset = self._run_phase(
                "C", rounds_c, batch_size,
                honest_host=False, train_host=True, train_players=True,
                stop_fn=None,
                log_interval=log_interval, spotlight_interval=spotlight_interval,
                save_dir=save_dir, rounds_offset=offset, t0=t0,
                csv_writer=writer, csv_file=csv_f, min_rounds=0,
            )

        print(f"\nCurriculum done. {offset} total rounds in {time.time()-t0:.1f}s")
        print(f"Log → {csv_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_log(e: dict) -> None:
    G, R, D, RST = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
    hr  = e["host_reward"]
    pr  = e["player_reward"]
    hc  = G if hr >= 0 else R
    pc  = G if pr >= 0 else R
    h_ent  = e.get("host_entropy",   float("nan"))
    p_ent  = e.get("player_entropy", float("nan"))
    h_loss = e.get("host_loss",      float("nan"))
    p_loss = e.get("player_loss",    float("nan"))
    print(
        f"[PPO][{e['round']:>8}]  "
        f"H={hc}{hr:+.3f}{RST}  P={pc}{pr:+.3f}{RST}  "
        f"wr={e['win_ratio']:.3f}  "
        f"hon={e['signal_honesty']:.2f}  fol={e['follow_rate']:.2f}  "
        f"{D}loss=({h_loss:.3f},{p_loss:.3f})  "
        f"ent=({h_ent:.3f},{p_ent:.3f}){RST}  "
        f"({e['elapsed_s']:.0f}s)"
    )


def _print_log_curriculum(e: dict) -> None:
    G, R, D, RST = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
    hr    = e["host_reward"]
    pr    = e["player_reward"]
    hc    = G if hr >= 0 else R
    pc    = G if pr >= 0 else R
    h_ent = e.get("host_entropy",   float("nan"))
    p_ent = e.get("player_entropy", float("nan"))
    h_loss = e.get("host_loss",     float("nan"))
    p_loss = e.get("player_loss",   float("nan"))
    phase  = e.get("phase", "?")
    print(
        f"[PPO][{phase}][{e['round']:>8}]  "
        f"H={hc}{hr:+.3f}{RST}  P={pc}{pr:+.3f}{RST}  "
        f"wr={e['win_ratio']:.3f}  "
        f"hon={e['signal_honesty']:.2f}  fol={e['follow_rate']:.2f}  "
        f"{D}loss=({h_loss:.3f},{p_loss:.3f})  "
        f"ent=({h_ent:.3f},{p_ent:.3f}){RST}  "
        f"({e['elapsed_s']:.0f}s)"
    )


def _save_checkpoints(runner: PPORunner, save_dir: str, rounds: int) -> None:
    torch.save(
        runner.host_agent.state_dict(),
        os.path.join(save_dir, f"host_{rounds}.pt"),
    )
    torch.save(
        runner.player_agent.state_dict(),
        os.path.join(save_dir, f"player_{rounds}.pt"),
    )
    print(f"  [ckpt] saved at round {rounds}")
