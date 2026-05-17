from doorl.env.payout import (
    calculate_payout,
    calculate_multiplier,
    host_pool_pnl,
)

__all__ = [
    "calculate_payout",
    "calculate_multiplier",
    "host_pool_pnl",
]


def __getattr__(name):  # pragma: no cover - lazy import for env classes
    if name in {"DooRLEnv", "parallel_env", "raw_env"}:
        from doorl.env import doorl_env

        return getattr(doorl_env, name)
    raise AttributeError(f"module 'doorl.env' has no attribute {name!r}")
