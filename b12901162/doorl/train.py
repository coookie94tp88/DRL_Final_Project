"""Training entry point for DooRL.

Usage:
    python -m doorl.train --config config/default.yaml --run-name myrun
    python -m doorl.train --config config/default.yaml --run-name short \\
        --override train.total_timesteps=200000 --override env.num_players=4
    python -m doorl.train --config config/default.yaml --run-name myrun \\
        --resume runs/myrun/ckpt/latest.pt
"""

from __future__ import annotations

import argparse
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from doorl import parallel_env
from doorl.checkpoint_utils import load_checkpoint, save_checkpoint
from doorl.config_utils import load_config
from doorl.happo import HAPPOTrainer, TrainConfig
from doorl.models import HostPolicy, HostPolicyConfig, PlayerPolicy, PlayerPolicyConfig
from doorl.training_progress import TrainingProgressTracker
from doorl.training_watch import run_training_watch


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_env_factory(cfg: Dict[str, Any]):
    env_cfg = cfg["env"]

    def factory():
        return parallel_env(**env_cfg)

    return factory


def _build_models(cfg: Dict[str, Any]):
    env = _build_env_factory(cfg)()
    p_cfg = PlayerPolicyConfig(
        obs_dim=env.player_obs_dim,
        num_players=env.num_players,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        nhead=cfg["model"]["nhead"],
        dim_ff=cfg["model"]["dim_ff"],
        dropout=cfg["model"]["dropout"],
        parameter_sharing=cfg["model"]["parameter_sharing"],
    )
    h_cfg = HostPolicyConfig(
        obs_dim=env.host_obs_dim,
        num_players=env.num_players,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        nhead=cfg["model"]["nhead"],
        dim_ff=cfg["model"]["dim_ff"],
        dropout=cfg["model"]["dropout"],
    )
    max_beta = float(cfg["train"].get("max_beta_concentration", 100.0))
    return PlayerPolicy(p_cfg, max_beta_concentration=max_beta), HostPolicy(h_cfg), env


def _build_train_cfg(cfg: Dict[str, Any]) -> TrainConfig:
    t = cfg["train"]
    return TrainConfig(
        lr=float(t["lr"]),
        gamma=float(t["gamma"]),
        gae_lambda=float(t["gae_lambda"]),
        clip_range=float(t["clip_range"]),
        clip_range_vf=float(t["clip_range_vf"]),
        ent_coef=float(t["ent_coef"]),
        vf_coef=float(t["vf_coef"]),
        grad_clip=float(t["grad_clip"]),
        target_kl=float(t["target_kl"]),
        n_steps=int(t["n_steps"]),
        n_epochs=int(t["n_epochs"]),
        minibatch_size=int(t.get("minibatch_size", 256)),
        num_envs=int(t.get("num_envs", 1)),
        total_timesteps=int(t["total_timesteps"]),
        reward_norm=bool(t.get("reward_norm", True)),
        agent_update_order=str(t.get("agent_update_order", "random")),
        warmup_steps=int(t.get("warmup_steps", 0)),
        log_interval=int(t.get("log_interval", 1)),
        seed=int(t.get("seed", 0)),
        adv_clip=float(t.get("adv_clip", 5.0)),
        reward_norm_clip=float(t.get("reward_norm_clip", 10.0)),
        log_ratio_clip=float(t.get("log_ratio_clip", 20.0)),
        max_beta_concentration=float(t.get("max_beta_concentration", 100.0)),
        anti_babbling=dict(t.get("anti_babbling", {})),
    )


class _StdoutLogger:
    def __init__(
        self,
        tb_dir: Path | None,
        progress: Optional[TrainingProgressTracker] = None,
    ) -> None:
        self.tb_dir = tb_dir
        self.progress = progress
        self.writer = None
        if tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(tb_dir))
            except Exception:
                self.writer = None

    def __call__(self, step: int, metrics: Dict[str, float]) -> None:
        prefix = ""
        if self.progress is not None:
            it = int(metrics.get("iter", 0))
            snap = self.progress.snapshot(step, it, metrics)
            prefix = self.progress.format_log_prefix(snap) + " | "
            if self.writer is not None:
                self.writer.add_scalar("progress/pct", snap.pct_complete, step)
                self.writer.add_scalar("progress/steps_per_sec", snap.steps_per_sec, step)
                if snap.eta_sec is not None:
                    self.writer.add_scalar(
                        "progress/eta_hours", snap.eta_sec / 3600.0, step
                    )
                self.writer.add_scalar(
                    "progress/health_ok", 1.0 if snap.health.level == "ok" else 0.0, step
                )

        loss_line = " | ".join(
            f"{k}={v:.4f}"
            for k, v in metrics.items()
            if k not in ("iter", "global_step")
        )
        print(f"[step {step}] {prefix}{loss_line}", flush=True)
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, float(v), step)


def _checkpoint_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("train", {})
    return {
        "interval_iters": int(t.get("checkpoint_interval_iters", 25)),
        "early_frac": float(t.get("checkpoint_early_frac", 0.10)),
        "mid_frac": float(t.get("checkpoint_mid_frac", 0.50)),
    }


def _watch_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("train", {})
    return {
        "interval_iters": int(t.get("watch_interval_iters", 50)),
        "max_rounds": int(t.get("watch_max_rounds", 8)),
        "style": str(t.get("watch_style", "compact")),
        "lang": str(t.get("watch_lang", "en")),
        "seed_base": int(t.get("watch_seed_base", 1000)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint (e.g. runs/myrun/ckpt/latest.pt) to continue training",
    )
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_name
    ckpt_dir = run_dir / "ckpt"
    tb_dir = run_dir / "tb_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    with open(run_dir / "git_hash.txt", "w") as f:
        f.write(_git_hash() + "\n")

    seed = int(cfg["train"].get("seed", 0))
    _set_seed(seed)

    player, host, _env = _build_models(cfg)
    tcfg = _build_train_cfg(cfg)
    factory = _build_env_factory(cfg)
    total_iters = max(1, tcfg.total_timesteps // tcfg.n_steps)

    start_iter = 0
    initial_step = 0

    trainer = HAPPOTrainer(
        env_factory=factory,
        player_policy=player,
        host_policy=host,
        cfg=tcfg,
        device=args.device,
        logger=None,
    )

    if args.resume:
        ck = load_checkpoint(args.resume, map_location=args.device)
        player.load_state_dict(ck["player"])
        host.load_state_dict(ck["host"])
        trainer._global_step = int(ck.get("global_step", 0))
        start_iter = int(ck.get("iteration", 0)) + 1
        initial_step = trainer._global_step
        print(
            f"resumed from {args.resume} at iter={start_iter} "
            f"global_step={trainer._global_step}",
            flush=True,
        )

    progress_tracker = TrainingProgressTracker(
        total_timesteps=tcfg.total_timesteps,
        total_iterations=total_iters,
        initial_step=initial_step,
        initial_iter=start_iter,
        target_kl=tcfg.target_kl,
        progress_file=run_dir / "progress.json",
    )
    trainer.logger = _StdoutLogger(tb_dir, progress=progress_tracker)

    if start_iter > 0:
        print(
            f"plan: resume at {100.0 * initial_step / tcfg.total_timesteps:.1f}% "
            f"({initial_step}/{tcfg.total_timesteps} steps), "
            f"{total_iters - start_iter} iterations remaining",
            flush=True,
        )
    else:
        print(
            f"plan: {total_iters} PPO iterations, {tcfg.total_timesteps} env steps, "
            f"n_steps={tcfg.n_steps}, device={args.device}",
            flush=True,
        )
    print(f"progress file: {run_dir / 'progress.json'}", flush=True)
    watch_settings = _watch_settings(cfg)
    watch_log = run_dir / "watch_log.txt"
    if watch_settings["interval_iters"] > 0:
        print(
            f"training watch: every {watch_settings['interval_iters']} iters, "
            f"{watch_settings['max_rounds']} rounds, style={watch_settings['style']} "
            f"(also {watch_log})",
            flush=True,
        )

    ck_settings = _checkpoint_settings(cfg)
    early_iter = max(0, int(total_iters * ck_settings["early_frac"]))
    mid_iter = max(0, int(total_iters * ck_settings["mid_frac"]))
    saved_milestones: set[str] = set()
    for tag in ("early", "mid"):
        if (ckpt_dir / f"{tag}.pt").exists():
            saved_milestones.add(tag)

    def on_iteration(it: int, global_step: int, _metrics: Dict[str, float]) -> None:
        nonlocal saved_milestones
        tag = None
        if it == early_iter and "early" not in saved_milestones:
            tag = "early"
            saved_milestones.add("early")
        elif it == mid_iter and "mid" not in saved_milestones:
            tag = "mid"
            saved_milestones.add("mid")
        elif it % ck_settings["interval_iters"] == 0 or it == total_iters - 1:
            tag = "latest"

        if tag is None:
            return

        path = ckpt_dir / f"{tag}.pt"
        save_checkpoint(
            path,
            player=player,
            host=host,
            config=cfg,
            global_step=global_step,
            iteration=it,
            tag=tag,
        )
        if tag == "latest" and it == total_iters - 1:
            save_checkpoint(
                ckpt_dir / "late.pt",
                player=player,
                host=host,
                config=cfg,
                global_step=global_step,
                iteration=it,
                tag="late",
            )
        print(f"saved checkpoint {path} (iter={it}, step={global_step})", flush=True)

        if (
            watch_settings["interval_iters"] > 0
            and it % watch_settings["interval_iters"] == 0
        ):
            run_training_watch(
                cfg,
                player,
                host,
                iteration=it,
                global_step=global_step,
                seed=watch_settings["seed_base"] + it,
                max_rounds=watch_settings["max_rounds"],
                style=watch_settings["style"],
                lang=watch_settings["lang"],
                log_path=watch_log,
            )

    trainer.train(start_iter=start_iter, on_iteration=on_iteration)

    save_checkpoint(
        ckpt_dir / "latest.pt",
        player=player,
        host=host,
        config=cfg,
        global_step=trainer._global_step,
        iteration=total_iters - 1,
        tag="latest",
    )
    save_checkpoint(
        ckpt_dir / "late.pt",
        player=player,
        host=host,
        config=cfg,
        global_step=trainer._global_step,
        iteration=total_iters - 1,
        tag="late",
    )
    final_pct = min(100.0, 100.0 * trainer._global_step / tcfg.total_timesteps)
    print(
        f"training complete ({final_pct:.1f}% of {tcfg.total_timesteps} steps); "
        f"checkpoints in {ckpt_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
