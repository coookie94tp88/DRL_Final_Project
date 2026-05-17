"""HAPPO-style trainer for DooRL.

Implements heterogeneous-agent PPO with:

* Independent per-agent rollouts and policies (shared encoder under the hood,
  but separate update calls per agent).
* Sequential trust-region updates in a random permutation each iteration
  (the HAPPO trick).
* Per-agent reward normalization and per-agent ``target_kl`` early stop.
* Anti-babbling hooks (off by default).

For simplicity v1 uses ``num_envs = 1`` parallel rollout; vectorization can be
added later without changing the API.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

from doorl.buffers import AgentTrajectory, RunningMeanStd, compute_gae
from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE, PHASE_HOST
from doorl.models.host_policy import HostPolicy
from doorl.models.transformer_policy import PlayerPolicy
from doorl.ppo_utils import (
    normalize_advantages,
    ppo_surrogate_loss,
    safe_optimizer_step,
)


# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    lr: float = 2.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    target_kl: float = 0.02
    n_steps: int = 2048
    n_epochs: int = 4
    minibatch_size: int = 256
    num_envs: int = 1
    total_timesteps: int = 200_000
    reward_norm: bool = True
    agent_update_order: str = "random"  # {random, fixed}
    warmup_steps: int = 0
    log_interval: int = 1
    seed: int = 0
    adv_clip: float = 5.0
    reward_norm_clip: float = 10.0
    log_ratio_clip: float = 20.0
    max_beta_concentration: float = 100.0

    # anti-babbling (read by trainer when wrapping policies)
    anti_babbling: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------


class HAPPOTrainer:
    def __init__(
        self,
        env_factory: Callable[[], Any],
        player_policy: PlayerPolicy,
        host_policy: HostPolicy,
        cfg: TrainConfig,
        device: str = "cpu",
        logger: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> None:
        self.env = env_factory()
        self.player_policy = player_policy.to(device)
        self.host_policy = host_policy.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.logger = logger

        self.player_opt = torch.optim.Adam(player_policy.parameters(), lr=cfg.lr)
        self.host_opt = torch.optim.Adam(host_policy.parameters(), lr=cfg.lr)

        self.num_players = self.env.num_players
        self.agent_names = ["host"] + [
            f"player_{i}" for i in range(self.num_players)
        ]
        self.player_names = self.agent_names[1:]

        self.reward_rms: Dict[str, RunningMeanStd] = {
            a: RunningMeanStd() for a in self.agent_names
        }
        self._global_step = 0

    # ------------------------- rollout -------------------------

    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, AgentTrajectory]:
        cfg = self.cfg
        env = self.env
        traj: Dict[str, AgentTrajectory] = {
            a: AgentTrajectory() for a in self.agent_names
        }
        obs, _ = env.reset()
        steps_taken = 0
        while steps_taken < cfg.n_steps:
            phase = env._phase
            actions, infos = self._build_step_actions(obs, phase)
            next_obs, rewards, terms, _, _ = env.step(actions)
            done = float(all(terms.values()))

            for a in self.agent_names:
                traj[a].add(
                    obs=obs[a],
                    action=actions[a],
                    logps=infos[a]["logp"],
                    value=float(infos[a]["value"]),
                    reward=float(rewards[a]),
                    done=done,
                    phase=phase,
                )
                if cfg.reward_norm:
                    self.reward_rms[a].update(np.array([rewards[a]]))

            obs = next_obs
            steps_taken += 1
            self._global_step += 1
            if done:
                obs, _ = env.reset()
        return traj

    def _build_step_actions(
        self, obs: Dict[str, np.ndarray], phase: int
    ) -> tuple[Dict[str, dict], Dict[str, dict]]:
        actions: Dict[str, dict] = {}
        infos: Dict[str, dict] = {}

        # host
        host_obs_t = torch.tensor(
            obs["host"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        host_action, host_info = self.host_policy.act(host_obs_t, deterministic=False)
        actions["host"] = {
            "public_door": int(host_action["public_door"][0]),
            "private_logits": host_action["private_logits"][0],
        }
        infos["host"] = {
            "logp": {"logp_public": host_info["logp_public"]},
            "value": float(host_info["value"][0]),
        }

        # players
        for i, name in enumerate(self.player_names):
            o_t = torch.tensor(
                obs[name], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            idx_t = torch.tensor([i], dtype=torch.long, device=self.device)
            act, info = self.player_policy.act(o_t, idx_t, phase=phase)
            actions[name] = {
                "bribe_pct": np.array(act["bribe_pct"][0], dtype=np.float32).reshape(
                    -1
                ),
                "door": int(act["door"][0]),
                "bet_pct": np.array(act["bet_pct"][0], dtype=np.float32).reshape(-1),
            }
            infos[name] = {
                "logp": {
                    "logp_bribe": info["logp_bribe"],
                    "logp_door": info["logp_door"],
                    "logp_bet": info["logp_bet"],
                },
                "value": float(info["value"][0]),
            }
        return actions, infos

    # ------------------------- update --------------------------

    def update(self, trajectories: Dict[str, AgentTrajectory]) -> Dict[str, float]:
        cfg = self.cfg
        order = list(self.agent_names)
        if cfg.agent_update_order == "random":
            random.shuffle(order)

        metrics: Dict[str, float] = {}
        for agent in order:
            t = trajectories[agent]
            if len(t) == 0:
                continue
            rewards = np.array(t.rewards, dtype=np.float64)
            if cfg.reward_norm:
                rewards = self.reward_rms[agent].normalize(
                    rewards, clip=cfg.reward_norm_clip
                )
            adv, returns = compute_gae(
                rewards.tolist(),
                t.values,
                t.dones,
                last_value=0.0,
                gamma=cfg.gamma,
                lam=cfg.gae_lambda,
            )
            adv = normalize_advantages(adv, clip=cfg.adv_clip)

            if agent == "host":
                m = self._update_host(t, adv, returns)
            else:
                idx = self.player_names.index(agent)
                m = self._update_player(idx, t, adv, returns)
            for k, v in m.items():
                metrics[f"{agent}/{k}"] = v
        return metrics

    def _tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _update_player(
        self,
        agent_idx: int,
        traj: AgentTrajectory,
        adv: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.cfg
        # Stack tensors
        obs = self._tensor(np.stack(traj.obs))
        idx_t = torch.full(
            (obs.size(0),), agent_idx, dtype=torch.long, device=self.device
        )
        bribe = self._tensor(
            np.array([a["bribe_pct"] for a in traj.actions]).reshape(-1)
        )
        door = self._tensor(
            np.array([a["door"] for a in traj.actions]), dtype=torch.long
        )
        bet = self._tensor(
            np.array([a["bet_pct"] for a in traj.actions]).reshape(-1)
        )
        actions = {"bribe_pct": bribe, "door": door, "bet_pct": bet}

        old_logp = (
            torch.cat([t["logp_bribe"].view(1) for t in traj.logp_components])
            + torch.cat([t["logp_door"].view(1) for t in traj.logp_components])
            + torch.cat([t["logp_bet"].view(1) for t in traj.logp_components])
        ).to(self.device)
        old_values = self._tensor(np.array(traj.values))
        adv_t = self._tensor(adv)
        ret_t = self._tensor(returns)

        N = obs.size(0)
        mb = max(1, min(cfg.minibatch_size, N))
        idx = np.arange(N)
        last_kl = 0.0
        last_loss = 0.0
        skipped = 0
        policy_backup = {
            k: v.detach().clone()
            for k, v in self.player_policy.state_dict().items()
        }
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb):
                sl = idx[start : start + mb]
                mb_idx = torch.as_tensor(sl, dtype=torch.long, device=self.device)
                sub_obs = obs[mb_idx]
                sub_idx_t = idx_t[mb_idx]
                sub_actions = {k: v[mb_idx] for k, v in actions.items()}
                ev = self.player_policy.evaluate(sub_obs, sub_idx_t, sub_actions)
                new_logp = ev["logp_bribe"] + ev["logp_door"] + ev["logp_bet"]
                policy_loss = ppo_surrogate_loss(
                    new_logp,
                    old_logp[mb_idx],
                    adv_t[mb_idx],
                    cfg.clip_range,
                    cfg.log_ratio_clip,
                )
                entropy = (
                    ev["entropy_bribe"].mean()
                    + ev["entropy_door"].mean()
                    + ev["entropy_bet"].mean()
                )
                value = ev["value"]
                v_clip = old_values[mb_idx] + torch.clamp(
                    value - old_values[mb_idx],
                    -cfg.clip_range_vf,
                    cfg.clip_range_vf,
                )
                v_loss1 = (value - ret_t[mb_idx]).pow(2)
                v_loss2 = (v_clip - ret_t[mb_idx]).pow(2)
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                ok, reason = safe_optimizer_step(
                    self.player_opt,
                    [self.player_policy],
                    loss,
                    cfg.grad_clip,
                )
                if not ok:
                    skipped += 1
                    self.player_policy.load_state_dict(policy_backup)
                    continue
                policy_backup = {
                    k: v.detach().clone()
                    for k, v in self.player_policy.state_dict().items()
                }

                with torch.no_grad():
                    kl = (old_logp[mb_idx] - new_logp).mean().item()
                    last_kl = kl
                    last_loss = float(loss.detach())
            if last_kl > cfg.target_kl:
                break
        out = {"loss": last_loss, "kl": last_kl}
        if skipped:
            out["skipped_steps"] = float(skipped)
        return out

    def _update_host(
        self, traj: AgentTrajectory, adv: np.ndarray, returns: np.ndarray
    ) -> Dict[str, float]:
        cfg = self.cfg
        obs = self._tensor(np.stack(traj.obs))
        pub = self._tensor(
            np.array([a["public_door"] for a in traj.actions]), dtype=torch.long
        )
        old_logp = torch.cat(
            [t["logp_public"].view(1) for t in traj.logp_components]
        ).to(self.device)
        old_values = self._tensor(np.array(traj.values))
        adv_t = self._tensor(adv)
        ret_t = self._tensor(returns)

        N = obs.size(0)
        mb = max(1, min(cfg.minibatch_size, N))
        idx = np.arange(N)
        last_kl = 0.0
        last_loss = 0.0
        skipped = 0
        host_backup = {
            k: v.detach().clone() for k, v in self.host_policy.state_dict().items()
        }
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb):
                sl = idx[start : start + mb]
                mb_idx = torch.as_tensor(sl, dtype=torch.long, device=self.device)
                sub_obs = obs[mb_idx]
                sub_actions = {"public_door": pub[mb_idx]}
                ev = self.host_policy.evaluate(sub_obs, sub_actions)
                new_logp = ev["logp_public"]
                policy_loss = ppo_surrogate_loss(
                    new_logp,
                    old_logp[mb_idx],
                    adv_t[mb_idx],
                    cfg.clip_range,
                    cfg.log_ratio_clip,
                )
                entropy = ev["entropy_public"].mean()
                # extra entropy bonus on per-player private distributions
                priv_logits = ev["private_logits"]
                priv_dist = Categorical(logits=priv_logits)
                priv_entropy = priv_dist.entropy().mean()
                ab = self.cfg.anti_babbling or {}
                ent_bonus = float(ab.get("host_entropy_bonus", 0.0))
                value = ev["value"]
                v_clip = old_values[mb_idx] + torch.clamp(
                    value - old_values[mb_idx],
                    -cfg.clip_range_vf,
                    cfg.clip_range_vf,
                )
                v_loss = 0.5 * torch.max(
                    (value - ret_t[mb_idx]).pow(2),
                    (v_clip - ret_t[mb_idx]).pow(2),
                ).mean()
                loss = (
                    policy_loss
                    + cfg.vf_coef * v_loss
                    - cfg.ent_coef * entropy
                    - ent_bonus * priv_entropy
                )
                ok, reason = safe_optimizer_step(
                    self.host_opt,
                    [self.host_policy],
                    loss,
                    cfg.grad_clip,
                )
                if not ok:
                    skipped += 1
                    self.host_policy.load_state_dict(host_backup)
                    continue
                host_backup = {
                    k: v.detach().clone()
                    for k, v in self.host_policy.state_dict().items()
                }
                with torch.no_grad():
                    last_kl = float((old_logp[mb_idx] - new_logp).mean().item())
                    last_loss = float(loss.detach())
            if last_kl > cfg.target_kl:
                break
        out = {"loss": last_loss, "kl": last_kl}
        if skipped:
            out["skipped_steps"] = float(skipped)
        return out

    # ------------------------- train loop --------------------

    def train(
        self,
        start_iter: int = 0,
        on_iteration: Optional[
            Callable[[int, int, Dict[str, float]], None]
        ] = None,
    ) -> int:
        cfg = self.cfg
        total_iters = max(1, cfg.total_timesteps // cfg.n_steps)
        for it in range(start_iter, total_iters):
            trajs = self.collect_rollout()
            metrics = self.update(trajs)
            if self.logger and (it % cfg.log_interval == 0):
                metrics["iter"] = float(it)
                metrics["global_step"] = float(self._global_step)
                self.logger(self._global_step, metrics)
            if on_iteration is not None:
                on_iteration(it, self._global_step, metrics)
        return self._global_step
