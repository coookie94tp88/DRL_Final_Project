"""Behavioral cloning (BC) pretrain for player policies.

Collects (obs, action) pairs from scripted experts (default: truthful Host +
greedy players that follow the public door), then fits the player policy with
supervised negative log-likelihood before HAPPO fine-tuning.

Usage:
    python -m doorl.bc_pretrain --config config/default.yaml --out runs/stage1_bc.pt \\
        --override env.num_players=4 --episodes 2000 --epochs 20

    python -m doorl.train --config config/default.yaml --run-name stage1_bc_rl \\
        --host-baseline truthful_host --init-players runs/stage1_bc.pt \\
        --override env.follow_public_bonus=2.0 ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from doorl import parallel_env
from doorl.baselines import GreedyPlayer, TruthfulHost
from doorl.config_utils import load_config
from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE, PHASE_HOST
from doorl.models import HostPolicy, PlayerPolicy
from doorl.train import _build_models, _set_seed


@dataclass
class BCSample:
    obs: np.ndarray
    agent_idx: int
    phase: int
    bribe_pct: float
    door: int
    bet_pct: float


def collect_bc_samples(
    cfg: Dict[str, Any],
    *,
    episodes: int = 500,
    seed: int = 0,
    bribe_pct: float = 0.05,
    bet_pct: float = 0.25,
) -> List[BCSample]:
    """Roll out truthful Host + greedy players; keep player decision steps."""
    env = parallel_env(**cfg["env"])
    host = TruthfulHost(env.num_players, seed=seed)
    player = GreedyPlayer(bribe_pct=bribe_pct, bet_pct=bet_pct)
    samples: List[BCSample] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            phase = env._phase
            if phase == PHASE_HOST:
                actions = {"host": host.act(env, obs)}
                for i in range(env.num_players):
                    actions[f"player_{i}"] = {
                        "bribe_pct": np.array([0.0], dtype=np.float32),
                        "door": 0,
                        "bet_pct": np.array([0.0], dtype=np.float32),
                    }
            else:
                actions = {"host": {"public_door": 0, "private_logits": np.zeros((env.num_players, 4))}}
                for i in range(env.num_players):
                    name = f"player_{i}"
                    if phase == PHASE_BRIBE:
                        a = player.act(env, obs, i, phase)
                        samples.append(
                            BCSample(
                                obs=obs[name].copy(),
                                agent_idx=i,
                                phase=PHASE_BRIBE,
                                bribe_pct=float(np.asarray(a["bribe_pct"]).reshape(-1)[0]),
                                door=0,
                                bet_pct=0.0,
                            )
                        )
                    elif phase == PHASE_BET:
                        a = player.act(env, obs, i, phase)
                        samples.append(
                            BCSample(
                                obs=obs[name].copy(),
                                agent_idx=i,
                                phase=PHASE_BET,
                                bribe_pct=0.0,
                                door=int(a["door"]),
                                bet_pct=float(np.asarray(a["bet_pct"]).reshape(-1)[0]),
                            )
                        )
                    actions[name] = a

            obs, _, terms, _, _ = env.step(actions)
            done = all(terms.values())

    return samples


def _batch_samples(
    samples: List[BCSample], indices: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
    obs = torch.tensor(
        np.stack([samples[i].obs for i in indices]), dtype=torch.float32, device=device
    )
    idx_t = torch.tensor(
        [samples[i].agent_idx for i in indices], dtype=torch.long, device=device
    )
    phases = torch.tensor(
        [samples[i].phase for i in indices], dtype=torch.long, device=device
    )
    bribe = torch.tensor(
        [samples[i].bribe_pct for i in indices], dtype=torch.float32, device=device
    )
    door = torch.tensor(
        [samples[i].door for i in indices], dtype=torch.long, device=device
    )
    bet = torch.tensor(
        [samples[i].bet_pct for i in indices], dtype=torch.float32, device=device
    )
    actions = {"bribe_pct": bribe, "door": door, "bet_pct": bet}
    return obs, idx_t, actions, phases


def train_bc(
    policy: PlayerPolicy,
    samples: List[BCSample],
    *,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
) -> List[float]:
    """Maximize log pi(a|s) on demonstration data (phase-aware)."""
    if not samples:
        raise ValueError("No BC samples collected")
    policy = policy.to(device)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(samples)
    losses: List[float] = []
    dev = torch.device(device)

    for _ in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            sl = perm[start : start + batch_size]
            obs, idx_t, actions, phases_t = _batch_samples(samples, sl, dev)
            ev = policy.evaluate(obs, idx_t, actions, phases=phases_t)
            logp = ev["logp"]
            # Door choice matters most for follow-public; up-weight BET steps.
            w = torch.where(phases_t == PHASE_BET, 3.0, 1.0)
            loss = -(logp * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    policy.eval()
    return losses


def save_player_ckpt(
    path: Path,
    player: PlayerPolicy,
    host: HostPolicy,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "player": player.state_dict(),
            "host": host.state_dict(),
            "config": cfg,
            "tag": "bc_pretrain",
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="BC pretrain player policy")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--out", default="runs/bc_pretrain.pt")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo-bribe-pct", type=float, default=0.05)
    p.add_argument("--demo-bet-pct", type=float, default=0.25)
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    _set_seed(args.seed)
    player, host, env_probe = _build_models(cfg)
    del env_probe

    print(
        f"collecting BC demos: {args.episodes} episodes, "
        f"truthful_host + greedy_players (bribe={args.demo_bribe_pct}, bet={args.demo_bet_pct})",
        flush=True,
    )
    samples = collect_bc_samples(
        cfg,
        episodes=args.episodes,
        seed=args.seed,
        bribe_pct=args.demo_bribe_pct,
        bet_pct=args.demo_bet_pct,
    )
    n_bribe = sum(1 for s in samples if s.phase == PHASE_BRIBE)
    n_bet = sum(1 for s in samples if s.phase == PHASE_BET)
    print(f"  samples: {len(samples)} ({n_bribe} bribe, {n_bet} bet)", flush=True)

    losses = train_bc(
        player,
        samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(f"BC done; final loss {losses[-1]:.4f} (first {losses[0]:.4f})", flush=True)

    out = Path(args.out)
    save_player_ckpt(out, player, host, cfg)
    print(f"saved player init → {out}", flush=True)
    print(
        "next: python -m doorl.train ... --host-baseline truthful_host "
        f"--init-players {out} --override env.follow_public_bonus=2.0",
        flush=True,
    )


if __name__ == "__main__":
    main()
