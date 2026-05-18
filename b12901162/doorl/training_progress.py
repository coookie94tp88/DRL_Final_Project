"""Progress tracking, ETA, and lightweight training-health hints."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h, rem = divmod(s, 3600)
    return f"{h}h {rem // 60}m"


@dataclass
class HealthStatus:
    level: str  # ok | warn | fail
    messages: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.level == "fail":
            return "FAIL: " + "; ".join(self.messages)
        if self.level == "warn":
            return "WARN: " + "; ".join(self.messages)
        return "OK"


def assess_training_health(
    metrics: Dict[str, float],
    *,
    target_kl: float = 0.02,
    loss_warn: float = 50.0,
) -> HealthStatus:
    """Heuristic checks on PPO metrics (optimization only, not game quality)."""
    messages: List[str] = []
    level = "ok"
    max_kl = 0.0
    max_loss = 0.0

    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
        fv = float(val)
        if math.isnan(fv) or math.isinf(fv):
            return HealthStatus("fail", [f"non-finite {key}"])
        if key.endswith("/kl"):
            max_kl = max(max_kl, fv)
        if key.endswith("/loss"):
            max_loss = max(max_loss, fv)

    if max_kl > target_kl * 10:
        messages.append(f"KL very high (max={max_kl:.3f}, target={target_kl})")
        level = "fail"
    elif max_kl > target_kl * 3:
        messages.append(f"KL elevated (max={max_kl:.3f})")
        level = "warn"

    if max_loss > loss_warn:
        messages.append(f"loss spike (max={max_loss:.1f})")
        if level == "ok":
            level = "warn"

    return HealthStatus(level, messages)


@dataclass
class ProgressSnapshot:
    global_step: int
    total_timesteps: int
    iteration: int
    total_iterations: int
    pct_complete: float
    elapsed_sec: float
    eta_sec: Optional[float]
    steps_per_sec: float
    health: HealthStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "total_timesteps": self.total_timesteps,
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "pct_complete": round(self.pct_complete, 2),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "eta_sec": round(self.eta_sec, 1) if self.eta_sec is not None else None,
            "steps_per_sec": round(self.steps_per_sec, 2),
            "health": self.health.level,
            "health_messages": self.health.messages,
        }


class TrainingProgressTracker:
    """Tracks % complete, ETA, and writes runs/<name>/progress.json."""

    def __init__(
        self,
        *,
        total_timesteps: int,
        total_iterations: int,
        initial_step: int = 0,
        initial_iter: int = 0,
        target_kl: float = 0.02,
        progress_file: Optional[Path] = None,
    ) -> None:
        self.total_timesteps = max(1, total_timesteps)
        self.total_iterations = max(1, total_iterations)
        self.initial_step = initial_step
        self.initial_iter = initial_iter
        self.target_kl = target_kl
        self.progress_file = progress_file
        self._t0 = time.monotonic()

    def snapshot(
        self,
        global_step: int,
        iteration: int,
        metrics: Dict[str, float],
    ) -> ProgressSnapshot:
        elapsed = time.monotonic() - self._t0
        steps_this_run = max(0, global_step - self.initial_step)
        pct = min(100.0, 100.0 * global_step / self.total_timesteps)
        steps_per_sec = steps_this_run / elapsed if elapsed > 1e-6 and steps_this_run > 0 else 0.0
        remaining = max(0, self.total_timesteps - global_step)
        eta_sec: Optional[float] = None
        if steps_per_sec > 1e-6 and remaining > 0:
            eta_sec = remaining / steps_per_sec

        health = assess_training_health(metrics, target_kl=self.target_kl)
        snap = ProgressSnapshot(
            global_step=global_step,
            total_timesteps=self.total_timesteps,
            iteration=iteration,
            total_iterations=self.total_iterations,
            pct_complete=pct,
            elapsed_sec=elapsed,
            eta_sec=eta_sec,
            steps_per_sec=steps_per_sec,
            health=health,
        )
        if self.progress_file is not None:
            self._write_progress_file(snap, metrics)
        return snap

    def _write_progress_file(
        self, snap: ProgressSnapshot, metrics: Dict[str, float]
    ) -> None:
        payload = {
            **snap.to_dict(),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        tmp = self.progress_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.progress_file)

    def format_log_prefix(self, snap: ProgressSnapshot) -> str:
        iter_pct = 100.0 * (snap.iteration + 1) / self.total_iterations
        eta_s = format_duration(snap.eta_sec) if snap.eta_sec is not None else "?"
        elapsed_s = format_duration(snap.elapsed_sec)
        return (
            f"{snap.pct_complete:.1f}% | "
            f"iter {snap.iteration + 1}/{self.total_iterations} ({iter_pct:.1f}%) | "
            f"elapsed {elapsed_s} | ETA {eta_s} | "
            f"{snap.steps_per_sec:.1f} steps/s | {snap.health.label}"
        )
