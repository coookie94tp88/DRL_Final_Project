"""DooRL: a MARL benchmark for endogenous bribery and signaling."""

__version__ = "0.1.0"


def parallel_env(**kwargs):  # pragma: no cover - thin re-export
    from doorl.env.doorl_env import parallel_env as _pe

    return _pe(**kwargs)


def raw_env(**kwargs):  # pragma: no cover - thin re-export
    from doorl.env.doorl_env import raw_env as _re

    return _re(**kwargs)
