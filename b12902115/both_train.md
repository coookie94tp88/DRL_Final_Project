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

## CLI

```bash
python train_both.py --num-doors 4 --seed 42 --total-bet-steps 120000 --save-every-episodes 50
# SAC entropy coeff (default 0.01; use 0 to disable entropy regularization in the loss):
python train_both.py --sac-alpha 0.01
```

Episode logs include **PubTruth** / **PrivTruth**: public honesty vs. private honesty **among bribers only** (`PrivTruth`).

**Rules:** bribe fraction $> 0$ ⇒ pay at least \$1 (if affordable); private signals only for `bribe > 0`; `BELIEVE_PRIVATE` without bribe maps to random.

## Payout threshold

Aligned with team default: **`payout_threshold = 0.10`** (10% break-even winning share).
