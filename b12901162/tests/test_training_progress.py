import math

from doorl.training_progress import (
    TrainingProgressTracker,
    assess_training_health,
    format_duration,
)


def test_format_duration():
    assert format_duration(45) == "45s"
    assert "m" in format_duration(125)


def test_assess_health_nan():
    h = assess_training_health({"host/loss": float("nan")})
    assert h.level == "fail"


def test_assess_health_ok():
    h = assess_training_health({"host/kl": 0.01, "host/loss": 1.0})
    assert h.level == "ok"


def test_progress_snapshot_eta():
    t = TrainingProgressTracker(
        total_timesteps=1000,
        total_iterations=10,
    )
    snap = t.snapshot(500, 4, {"host/kl": 0.01, "host/loss": 1.0})
    assert snap.pct_complete == 50.0
    assert "50.0%" in t.format_log_prefix(snap)
