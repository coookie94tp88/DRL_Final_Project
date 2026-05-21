"""Shared observation flattening / encoding for player Transformer agents."""

import torch

CURR_DIM = 4
HIST_DIM = 12
SEQ_LEN = 50
STATE_DIM = CURR_DIM + SEQ_LEN * HIST_DIM  # 604


def flatten_obs(obs_dict, device="cpu"):
    """Flatten player obs dict to (N, STATE_DIM)."""
    curr = torch.FloatTensor(obs_dict["current"]).to(device)
    hist = torch.FloatTensor(obs_dict["history"]).to(device)
    hist_flat = hist.view(hist.shape[0], -1)
    return torch.cat([curr, hist_flat], dim=1)


def encode_features(state_flat, seq_len=SEQ_LEN):
    """Split flat state; all features are already continuous scalars."""
    curr = state_flat[:, :CURR_DIM]
    hist = state_flat[:, CURR_DIM:].view(state_flat.shape[0], seq_len, HIST_DIM)
    return curr, hist
