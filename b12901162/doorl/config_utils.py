"""Config loading + CLI override helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form ``a.b.c=value`` (parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"--override expects key=value, got {ov!r}")
        key, value = ov.split("=", 1)
        parsed: Any
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = parsed
    return cfg


def load_config(
    config_path: str | os.PathLike, overrides: List[str] | None = None
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    base = cfg.pop("base", None)
    if base is not None:
        base_path = Path(config_path).parent / base
        base_cfg = load_yaml(base_path)
        cfg = deep_merge(base_cfg, cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
