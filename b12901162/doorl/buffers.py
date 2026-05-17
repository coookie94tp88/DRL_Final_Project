"""Rollout buffers and running reward normalization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


class RunningMeanStd:
    """Welford-style running mean/std over a scalar stream."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = math.sqrt(self.var + 1e-8)
        y = (np.asarray(x, dtype=np.float64) - self.mean) / std
        if clip > 0:
            y = np.clip(y, -clip, clip)
        return y


@dataclass
class AgentTrajectory:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    logp_components: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    phases: List[int] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: dict,
        logps: Dict[str, torch.Tensor],
        value: float,
        reward: float,
        done: float,
        phase: int,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.logp_components.append(logps)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.phases.append(phase)

    def __len__(self) -> int:
        return len(self.obs)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[float],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + np.asarray(values, dtype=np.float64)
    return adv, returns
