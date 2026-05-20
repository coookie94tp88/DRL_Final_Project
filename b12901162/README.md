# DooRL — a MARL benchmark for endogenous bribery and signaling

DooRL (4-Door RL) is a multi-agent reinforcement-learning environment in which
`N` players bet on one of four doors, while a central **Host** (the Oracle) holds
the ground truth and may sell hints. Each round players first choose a *bribe
percentage* of their balance; the Host then emits a public and per-player
private signal; players finally pick a door and a bet percentage. Payouts use a
parimutuel-style dynamic-odds rule that lets the Host profit when the winning
bet share is below a configurable threshold `tau`.

See [`spec.md`](spec.md) for the full design, hypotheses (H1–H4), and risk
register.

**Training, TensorBoard, checkpoints, resume, eval, BC pretrain, and reward shaping:** see **[`TRAINING.md`](TRAINING.md)**.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Smoke run (tiny env, ~30s on CPU)
python -m doorl.train --config config/default.yaml --run-name smoke \
    --override env.num_players=4 --override env.max_rounds=20 \
    --override train.total_timesteps=2048 --override train.n_steps=512 \
    --override model.d_model=64

# Full training (see TRAINING.md for GPU, checkpoints, TensorBoard, resume)
python -m doorl.train --config config/default.yaml --run-name full

# Evaluate a checkpoint
python -m doorl.eval --ckpt runs/full/ckpt/latest.pt --episodes 500

# Watch one episode (fancy terminal UI)
python -m doorl.watch --ckpt runs/full/ckpt/latest.pt --seed 0

# Evaluate a scripted baseline (no checkpoint needed)
python -m doorl.eval --config config/default.yaml --baseline truthful_host \
    --episodes 50 --override env.num_players=4
```

Available baselines: `random_host`, `truthful_host`, `noisy_truthful_host`,
`greedy_players`, `no_bribe_players`.

## Sweep configs

Six sweep templates live in `config/`. Use them with your own driver script;
each one inherits `default.yaml` and lists the values to vary:

| File | Knob varied |
| --- | --- |
| `sweep_payout_threshold.yaml` | `env.payout_threshold` ∈ {0.10, 0.20, 0.30} |
| `sweep_multiplier_cap.yaml` | `env.max_multiplier` ∈ {20, 50, 100, null} |
| `sweep_num_players.yaml` | `env.num_players` ∈ {4, 10, 20} |
| `sweep_host_mode.yaml` | host: {learned, scripted_truthful, scripted_noisy} |
| `sweep_param_sharing.yaml` | `model.parameter_sharing` ∈ {encoder, none, full} |
| `sweep_balance_visibility.yaml` | `env.balance_visibility` ∈ {full, own_only, noisy} |

## Anti-babbling trigger

The default training run uses a uniform initial Host policy. In signaling games
the trivial **babbling equilibrium** (Host emits noise, players ignore signals)
is a known failure mode. If after about `1e6` env steps the metric
`mi_private_truth` does not exceed the random baseline by more than ~0.05 bits,
turn on the curriculum:

```yaml
train:
  anti_babbling:
    enabled: true
    init_host_toward_truth: 2.0   # logit bias on true door at init
    host_entropy_bonus: 0.01      # extra entropy coef on Host private heads
    bribery_floor: 0.01           # minimum bribe_pct during warm-up
    warmup_steps: 500000
```

and rerun.

## Repository layout

```
spec.md                          # design spec; the single source of truth
config/                          # YAML configs + 6 sweeps
doorl/
  env/doorl_env.py               # DooRLEnv (PettingZoo ParallelEnv)
  env/payout.py                  # tau-parameterized payout math
  env/types.py
  models/transformer_policy.py   # player policies (3 parameter_sharing modes)
  models/host_policy.py          # host policy (internal private-head sharing)
  happo.py                       # HAPPO trainer
  buffers.py                     # rollout buffer + running reward norm
  metrics.py                     # MI estimators + acceptance targets
  baselines/                     # 5 scripted baselines
  train.py                       # CLI training entry point
  eval.py                        # CLI evaluation + cross-play entry point
tests/                           # pytest suite (24 tests)
TRAINING.md                      # training, TensorBoard, checkpoints, eval
```

## Run the tests

```bash
python -m pytest -q
```

## Cite

If you use DooRL in published research, please cite the project page.
