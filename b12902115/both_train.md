# Co-Training (`train_both.py`) — Belief Edition

This document complements [README.md](README.md). The **environment** uses belief actions (`pub` / `priv` / `rnd`), not direct door picks. Training still uses the same two-step SAC + Host RDQN stack from `b13902055`, with belief-adapted reward shaping.

## Player (SAC)

- **State:** flat vector `STATE_DIM = 604` (`obs_encoding.py`) — no door one-hot in observations.
- **Actors:** `BribeActor` (continuous) + `BetActor` (3-way Gumbel-Softmax + bet fraction).
- **Shaped rewards (in training loop, not env):**
  - Trust-profit bribe rebate / follow-private bonus
  - Penalty for following private when it lied
  - Crowding penalty from mapped-door bet share
  - Diversity bonus for less crowded doors

## Host (RDQN)

- **Shaped reward:** env host profit + `2.0 × bribes` + truth bonuses − lie penalty (bribe-weighted).
- Checkpoints: `checkpoints/host.pth` with `num_doors` metadata.

## SB3 baseline (`train_sb3.py`)

Alternative co-training with **Stable-Baselines3 PPO** (same belief env as `env.py`):

- **Player:** `MultiInputPolicy` + `PlayerExtractor` (flatten `current` + `history`)
- **Action (flat Box, 3×N):** sub-step A uses `[:N]` bribe only; sub-step B uses `[N:2N]` bet + `[2N:3N]` belief (after SIGNAL — same as `eval_sb3`)
- **Belief encoding:** continuous dims in $[0,2]$ → `rint` → `0=pub, 1=priv, 2=rnd` (see `train_sb3.py` docstring)
- **Timesteps:** `--total-rounds R` → `2×R` PPO env steps (two steps per game round)
- **Host:** MLP policy gradient on env `host_profit` (includes `private_honesty_hist`)
- **Checkpoints:** `checkpoints_sb3/player_model_<round>.zip`, `host_model_<round>.pt`
- **Loader:** `sb3_player_agent.SB3PlayerAgent` for eval scripts

```bash
conda activate doorl
export MPLCONFIGDIR=/tmp2/b12902115/tmp/mpl
python train_sb3.py --total-rounds 2000 --save-every 200 --num-doors 4 --max-rounds 15
python train_sb3.py --resume-player checkpoints_sb3/player_model_800.zip --resume-host checkpoints_sb3/host_model_800.pt
```

## CLI (SAC + RDQN)

```bash
python train_both.py --num-doors 4 --seed 42 --total-bet-steps 120000 --save-every-episodes 50
# SAC entropy coeff (default 0.01; use 0 to disable entropy regularization in the loss):
python train_both.py --sac-alpha 0.01
```

Episode logs include **PubTruth** / **PrivTruth**: public honesty vs. private honesty **among bribers only** (`PrivTruth`).

**Rules:** bribe fraction $> 0$ ⇒ pay at least \$1 (if affordable); private signals only for `bribe > 0`; `BELIEVE_PRIVATE` without bribe maps to random.

## Payout threshold

Aligned with team default: **`payout_threshold = 0.10`** (10% break-even winning share).
