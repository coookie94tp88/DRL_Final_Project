"""Save/load training checkpoints for DooRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from doorl.models import HostPolicy, PlayerPolicy


def save_checkpoint(
    path: Path | str,
    *,
    player: PlayerPolicy,
    host: HostPolicy,
    config: Dict[str, Any],
    global_step: int,
    iteration: int,
    tag: str = "latest",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "player": player.state_dict(),
        "host": host.state_dict(),
        "config": config,
        "global_step": int(global_step),
        "iteration": int(iteration),
        "tag": tag,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_policies_from_checkpoints(
    player_ckpt: Optional[str],
    host_ckpt: Optional[str],
    default_ckpt: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load config and state dicts, allowing host and player from different files."""
    if default_ckpt is not None and player_ckpt is None and host_ckpt is None:
        ck = load_checkpoint(default_ckpt)
        return ck["config"], ck["player"], ck["host"]

    cfg: Optional[Dict[str, Any]] = None
    player_sd = None
    host_sd = None

    if player_ckpt is not None:
        ck_p = load_checkpoint(player_ckpt)
        cfg = ck_p["config"]
        player_sd = ck_p["player"]
    if host_ckpt is not None:
        ck_h = load_checkpoint(host_ckpt)
        cfg = cfg or ck_h["config"]
        host_sd = ck_h["host"]

    if cfg is None:
        raise ValueError("Provide --ckpt, --player-ckpt, or --host-ckpt")
    if player_sd is None or host_sd is None:
        fallback = default_ckpt or player_ckpt or host_ckpt
        if fallback is None:
            raise ValueError("Need both player and host weights; pass --ckpt or both split ckpts")
        ck = load_checkpoint(fallback)
        player_sd = player_sd or ck["player"]
        host_sd = host_sd or ck["host"]
        cfg = cfg or ck["config"]

    return cfg, player_sd, host_sd
