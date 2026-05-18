# DooRL training guide

This document covers how to **train**, **monitor progress** (TensorBoard), **save and resume** checkpoints, and **evaluate** policies—including cross-play between early and late training.

For environment design and hypotheses, see [`spec.md`](spec.md). For install and repo layout, see [`README.md`](README.md).

All commands assume you are in **your** project folder:

```bash
cd /path/to/DRL_Final_Project/b12901162
pip install -r requirements.txt
```

### If an earlier run looks useless

An env bug (true door did not rotate between rounds) meant **`learn_4p` checkpoints trained the wrong game**. That is fixed now; **start a new run** (e.g. `learn_4p_v2`). Do not resume old `learn_4p` weights.

**During training** you now get a live mini-game log every `train.watch_interval_iters` (default 50) in the terminal and in `runs/<run-name>/watch_log.txt` — you do not have to guess from loss alone.

---

## 1. Start a training run

### Smoke test (~30 seconds, CPU)

Verifies the pipeline before a long run:

```bash
python -m doorl.train --config config/default.yaml --run-name smoke \
  --device cpu \
  --override env.num_players=4 --override env.max_rounds=20 \
  --override train.total_timesteps=2048 --override train.n_steps=512 \
  --override model.d_model=64
```

### Recommended 4-player run (GPU)

Default config uses 10 players and 20M timesteps—too heavy for a course project. A stable 4-player setup:

```bash
python -m doorl.train --config config/default.yaml --run-name learn_4p_v2 \
  --device cuda \
  --override env.num_players=4 \
  --override env.max_rounds=50 \
  --override train.total_timesteps=2000000 \
  --override train.n_steps=2048 \
  --override train.lr=3e-5 \
  --override train.n_epochs=2 \
  --override train.grad_clip=0.3 \
  --override train.clip_range=0.1
```

### See the game **while** training (built in)

Every `train.watch_interval_iters` (default **50** PPO iterations), training prints a short episode using **current** weights:

```text
════ TRAINING WATCH  iter=50  step=102400  seed=1050  rounds=8 ════
  R1: true=D pub=C✗ x=0.31 host=+120 alive=4/4 priv_ok=2/4 picks=[ABBC]
  R2: true=A pub=A✓ x=0.12 host=+80  alive=4/4 priv_ok=1/4 picks=[CDAA]
  ...
════ END TRAINING WATCH ════
```

Same text is appended to `runs/<run-name>/watch_log.txt` (scroll back anytime).

| Override | Effect |
| --- | --- |
| `train.watch_interval_iters=25` | More frequent snapshots |
| `train.watch_interval_iters=0` | Turn off |
| `train.watch_style=fancy` | Full bars + table (slower, noisy) |
| `train.watch_style=plain` | Chinese emoji blocks |
| `train.watch_lang=zh` | Chinese compact lines |

| Setting | Why |
| --- | --- |
| `total_timesteps=2000000` | ~2M steps is a reasonable target for 4 players |
| `lr=5e-5`, `n_epochs=2` | Reduces NaN / Beta-policy blow-ups vs default |
| `grad_clip=0.3`, `clip_range=0.1` | Tighter trust region when rewards are spiky |
| `--device cuda` | Much faster; use `nvidia-smi` to confirm GPU use |

**CPU vs GPU:** Default CLI device is `cpu`. Always pass `--device cuda` for long runs if you have a GPU.

### Override any config field

```bash
python -m doorl.train --config config/default.yaml --run-name myrun \
  --override train.checkpoint_interval_iters=10
```

---

## 2. What gets written to disk

Each run creates `runs/<run-name>/`:

```
runs/<run-name>/
  config.yaml          # frozen config for this run
  git_hash.txt         # commit at train start
  tb_logs/             # TensorBoard event files
  ckpt/
    latest.pt          # overwritten every N PPO iterations
    early.pt           # once at ~10% of training
    mid.pt             # once at ~50%
    late.pt            # at the final iteration
```

Checkpoint files contain `player`, `host`, `config`, `global_step`, and `iteration`.

### Checkpoint schedule (defaults in `config/default.yaml`)

| File | When |
| --- | --- |
| `latest.pt` | Every `train.checkpoint_interval_iters` (default **25**) PPO iterations |
| `early.pt` | Once at `train.checkpoint_early_frac` (default **10%**) of total iterations |
| `mid.pt` | Once at `train.checkpoint_mid_frac` (default **50%**) |
| `late.pt` | End of training |

With `n_steps=2048`, one PPO iteration ≈ **2048 environment steps**. So `latest.pt` updates about every **25 × 2048 ≈ 51,200** steps by default.

**Important:** Only runs started with the current `doorl/train.py` write periodic checkpoints. If `ckpt/` is empty while training, the process was started with an older version that saved only at the end—restart training with the current code to get mid-run saves.

---

## 3. Stop training and resume later

### When it is safe to stop

Wait until the log prints something like:

```text
saved checkpoint runs/<run-name>/ckpt/latest.pt (iter=25, step=51200)
```

If you stop **before** any checkpoint exists, you **cannot** resume and must train from scratch.

### Resume command

Use the **same** `--override` flags as the original run (or rely on the config stored inside the checkpoint):

```bash
python -m doorl.train --config config/default.yaml --run-name learn_4p \
  --device cuda \
  --resume runs/learn_4p/ckpt/latest.pt \
  --override env.num_players=4 \
  --override env.max_rounds=50 \
  --override train.total_timesteps=2000000 \
  --override train.n_steps=2048 \
  --override train.lr=5e-5 \
  --override train.n_epochs=2 \
  --override train.grad_clip=0.3 \
  --override train.clip_range=0.1
```

Resume loads policy weights and continues from `iteration + 1` and `global_step` stored in the checkpoint. Optimizer state is re-initialized (acceptable for continued learning).

### Manual snapshot (optional)

Before a risky stop, copy the latest file:

```bash
cp runs/learn_4p/ckpt/latest.pt runs/learn_4p/ckpt/snapshot_step_100k.pt
```

---

## 4. Progress %, ETA, and health (built into training logs)

Every log line now includes a **progress header** before the loss metrics:

```text
[step 174080] 8.7% | iter 85/976 (8.7%) | elapsed 1h 58m | ETA 20h 45m | 24.3 steps/s | OK | host/loss=0.6145 | ...
```

| Field | Meaning |
| --- | --- |
| `8.7%` | `global_step / train.total_timesteps` |
| `iter 85/976` | PPO iteration (rollout + update cycles) |
| `elapsed` | Wall time since this train process started (reset on resume) |
| `ETA` | Estimated time to reach `total_timesteps` from current speed |
| `steps/s` | Throughput since start/resume |
| `OK` / `WARN` / `FAIL` | Quick optimization health (see below) |

### `progress.json` (live status file)

Updated every log step at `runs/<run-name>/progress.json`:

```bash
cat runs/learn_4p/progress.json
```

Useful for scripts or checking progress without parsing the terminal. Fields include `pct_complete`, `eta_sec`, `steps_per_sec`, `health`, and the latest PPO metrics.

### TensorBoard progress scalars

| Tag | Meaning |
| --- | --- |
| `progress/pct` | % of `total_timesteps` |
| `progress/steps_per_sec` | Throughput |
| `progress/eta_hours` | Estimated hours remaining |
| `progress/health_ok` | 1 if optimization health is OK, else 0 |

---

## 5. Monitor training with TensorBoard

Training logs scalars every `train.log_interval` PPO iterations (default: every iteration) to `runs/<run-name>/tb_logs/`.

### Start TensorBoard

```bash
tensorboard --logdir runs/<run-name>/tb_logs --port 6006
```

Open **http://localhost:6006** in a browser.

Compare multiple runs:

```bash
tensorboard --logdir runs --port 6006
```

### WSL / remote machine

If the browser runs on Windows and training on WSL, forward the port (PowerShell or CMD on Windows):

```bash
ssh -L 6006:localhost:6006 user@wsl-host
```

Or use VS Code / Cursor port forwarding for port `6006`.

### Scalars you will see

| Tag | Meaning |
| --- | --- |
| `host/loss`, `player_0/loss`, … | PPO surrogate loss per agent (update quality) |
| `host/kl`, `player_*/kl` | Approximate KL after update; compare to `train.target_kl` (0.02) |
| `global_step` | Total environment steps so far |
| `iter` | PPO iteration index |

TensorBoard does **not** log game-level metrics (bankruptcy, mutual information, host profit). Use **evaluation** (section 6) for those.

### Reading curves: healthy vs problematic

**Healthy (optimization running):**

- Losses trend down or plateau after an initial drop
- KL usually below ~0.05; occasional spikes are normal
- No `nan` in the terminal log

**Problematic:**

- Loss or KL explodes, or training prints NaN (common with default `lr` on this env—lower `lr`, fewer `n_epochs`)
- KL stays very high every step → policies changing too fast; lower `lr` or `clip_range`

**Healthy losses ≠ solved task.** Low PPO loss does not mean players exploit signals or that the Host is profitable. Always run `doorl.eval` on checkpoints for behavior.

### Progress without TensorBoard

Training also prints lines to stdout:

```text
[step 174080] player_0/loss=... | host/loss=... | host/kl=... | iter=85.0000 | global_step=174080.0000
```

Rough ETA for 2M steps with `n_steps=2048`:

- Total PPO iterations ≈ `total_timesteps / n_steps` (e.g. 2_000_000 / 2048 ≈ **976**)
- Wall time depends on GPU, `num_players`, `max_rounds`; expect **many hours** on GPU for 2M steps

---

## 6. Evaluate during or after training

### Single checkpoint

```bash
python -m doorl.eval --ckpt runs/learn_4p_v2/ckpt/latest.pt --episodes 100
```

Default output is a **Chinese quick-judgment report** (✓/✗ lines + overall 🔴/🟡/🟢). Raw JSON: add `--json`. English: `--lang en`.

### Watch one episode in the terminal (see behavior)

OracleGambit-style bars and per-player table:

```bash
python -m doorl.watch --ckpt runs/learn_4p/ckpt/latest.pt --seed 0 \
  --override env.num_players=4 --override env.max_rounds=20
```

Plain Chinese block (like your old round 結算 log):

```bash
python -m doorl.watch --ckpt runs/learn_4p/ckpt/latest.pt --style plain --lang zh \
  --override env.num_players=10 --max-rounds 15
```

Same via eval:

```bash
python -m doorl.eval --ckpt runs/learn_4p/ckpt/latest.pt --watch --watch-style fancy
```

Compare baselines:

```bash
python -m doorl.watch --baseline truthful_host --seed 42 --override env.num_players=5
```

### Scripted baseline (no checkpoint)

```bash
python -m doorl.eval --config config/default.yaml \
  --baseline truthful_host --episodes 50 \
  --override env.num_players=4
```

Baselines: `random_host`, `truthful_host`, `noisy_truthful_host`, `greedy_players`, `no_bribe_players`.

### Metrics to watch (behavior)

| Metric | Rough goal |
| --- | --- |
| `bankruptcy_rate` | Lower than untrained / random baselines |
| `median_x` | Near `env.payout_threshold` (τ, default 0.20) |
| `mi_private_truth` | Rises if players use private hints |
| `mi_public_truth` | Rises if public signal is informative |
| `public_true_private_false_rate` | Lower is less deceptive Host |
| `mean_R_host` | Positive if Host is extracting edge |

Run eval every few hundred thousand steps on `latest.pt` while training continues in another terminal.

---

## 7. Cross-play (early vs late policies)

Compare a **late Host** against **early players**, or the reverse, using split checkpoints:

```bash
# Late host vs early players
python -m doorl.eval \
  --host-ckpt runs/learn_4p/ckpt/late.pt \
  --player-ckpt runs/learn_4p/ckpt/early.pt \
  --episodes 200

# Early host vs late players
python -m doorl.eval \
  --host-ckpt runs/learn_4p/ckpt/early.pt \
  --player-ckpt runs/learn_4p/ckpt/late.pt \
  --episodes 200
```

You can also mix `mid.pt`, `latest.pt`, or manual snapshots. Output JSON includes `config.cross_play: true` when host and player come from different files.

---

## 8. How to tell if training is failing (decision guide)

Use **two layers**: optimization health (automatic) vs **task** quality (manual eval).

### Layer A — Optimization (automatic `OK` / `WARN` / `FAIL`)

| Signal | Likely meaning | Action |
| --- | --- | --- |
| `FAIL: non-finite ...` | NaN / crash imminent | Stop; lower `lr`, `n_epochs`; see troubleshooting |
| `FAIL: KL very high` | Policy updates too aggressive | Lower `lr` or `clip_range`; reduce `n_epochs` |
| `WARN: KL elevated` | Noisy but may recover | Watch 20+ iterations; if persistent, tune hyperparams |
| `WARN: loss spike` | Large PPO loss | Often transient; fail only if repeated + eval degrades |
| `OK` | Gradients stable | **Does not mean** the game is learned |

### Layer B — Task / behavior (you must run eval)

Run every **~100k–250k** steps (or when `early.pt` / `mid.pt` appear):

```bash
python -m doorl.eval --ckpt runs/<run>/ckpt/latest.pt --episodes 100
```

Compare to a **baseline** at the same `num_players`:

```bash
python -m doorl.eval --config config/default.yaml \
  --baseline truthful_host --episodes 100 --override env.num_players=4
```

| Eval metric | Training likely **working** | Training likely **stuck / failing** |
| --- | --- | --- |
| `bankruptcy_rate` | Falling vs early checkpoints / baselines | Stays very high (>0.4) after 500k+ steps |
| `mi_private_truth` | Rising above ~0.05 vs random | Flat near 0 after ~1M steps → try anti-babbling |
| `median_x` | Moving toward τ (0.20) | Stuck >> τ (e.g. 0.7+) — pool not competitive |
| `mean_R_host` | Reasonable, not exploding | Wild swings or always deeply negative |
| `public_true_private_false_rate` | Stable or decreasing | ~1.0 forever — deceptive host, players ignore hints |

### Layer C — Practical workflow (掌握进度)

1. **Glance at terminal** — % and ETA; confirm `steps/s` is stable (GPU >> CPU).
2. **TensorBoard** — loss down/plateau, KL not permanently high; plot `progress/pct`.
3. **`progress.json`** — quick read without scrolling logs.
4. **Periodic eval** — save JSON outputs (`eval_step_200k.json`) to compare over time.
5. **Cross-play** at `early` vs `late` — see if Host/Players co-evolved.
6. **Stop early only if** — repeated `FAIL`, or eval flat after your target budget (e.g. 1M steps with no MI gain).

**Do not stop** just because loss is low: in signaling games, **babbling** (low loss, useless signals) is common. MI and bankruptcy in eval are the real pass/fail for your project hypotheses.

---

## 9. Recovering from a NaN crash

If training dies with `ValueError: ... Beta ... nan`:

1. **Do not** continue from a checkpoint saved *after* the crash (weights may be corrupted).
2. Use the **last good** checkpoint, e.g. `runs/learn_4p/ckpt/latest.pt` from the log line before the traceback.
3. **Resume** with the same overrides plus stability knobs (now defaults in `config/default.yaml`):

```bash
python -m doorl.train --config config/default.yaml --run-name learn_4p \
  --device cuda \
  --resume runs/learn_4p/ckpt/latest.pt \
  --override env.num_players=4 --override env.max_rounds=50 \
  --override train.total_timesteps=2000000 --override train.n_steps=2048 \
  --override train.lr=3e-5 --override train.n_epochs=2 \
  --override train.grad_clip=0.3 --override train.clip_range=0.1 \
  --override train.adv_clip=5.0 --override train.reward_norm_clip=10.0
```

The trainer now **clips advantages and reward norms**, **caps Beta concentrations**, **clamps PPO log-ratios**, and **skips bad minibatches** while rolling back policy weights. Watch for `skipped_steps` in logs; many skips in a row means lower `lr` again.

---

## 10. Troubleshooting

| Problem | What to do |
| --- | --- |
| `ckpt/` empty after hours | Stop and restart with current `train.py`; old runs only saved at end |
| NaN in Beta policy / crash | Resume last good `latest.pt`; see §9; try `lr=3e-5`, `adv_clip=5`, `reward_norm_clip=10` |
| Training very slow | `--device cuda`; reduce `env.max_rounds` or `num_players` for debugging |
| `mi_private_truth` flat after ~1M steps | Enable `train.anti_babbling` in config (see README anti-babbling section) |
| Tests fail | Run from project root: `python -m pytest tests/ -q` |
| TensorBoard empty | Confirm `runs/<run>/tb_logs/events.out.tfevents.*` exists and grows during training |

---

## 11. Config knobs reference

Relevant `train:` keys in `config/default.yaml`:

```yaml
train:
  total_timesteps: 20000000
  n_steps: 2048              # rollout length per PPO iteration
  lr: 2.0e-4
  n_epochs: 4
  target_kl: 0.02
  log_interval: 1            # how often to log to stdout + TensorBoard
  checkpoint_interval_iters: 25
  checkpoint_early_frac: 0.10
  checkpoint_mid_frac: 0.50
```

Relevant `env:` keys for training cost and difficulty:

```yaml
env:
  num_players: 10
  max_rounds: 200
  payout_threshold: 0.20     # tau
```

---

## 12. Quick checklist

1. [ ] Smoke run on CPU completes and writes `runs/smoke/ckpt/latest.pt`
2. [ ] Start long run with `--device cuda` and a unique `--run-name`
3. [ ] Watch log lines for `%`, `ETA`, and `OK`/`WARN`/`FAIL`
4. [ ] Open TensorBoard (`progress/pct`, losses, KL)
5. [ ] Optionally `cat runs/<run-name>/progress.json` from another terminal
6. [ ] Confirm `saved checkpoint ... latest.pt` before stop/resume
7. [ ] Periodically run `doorl.eval` on `latest.pt` and save JSON for comparison
8. [ ] After training, run cross-play with `early.pt` vs `late.pt`
9. [ ] Keep `runs/<run-name>/config.yaml` with results for reproducibility
