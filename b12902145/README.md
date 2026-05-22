# OracleGambit: Strategic Information Manipulation in Multi-Agent RL

**OracleGambit** is a Multi-Agent Reinforcement Learning (MARL) environment that studies the emergence of deceptive signaling and trust dynamics. A single **Host** agent knows the ground truth and must decide whether to guide or deceive **N Player** agents, while Players must learn when to trust the Host's signal.

> Submitted to **NTU-DRL-MiniConf 2026**

---

## Table of Contents

1. [Game Overview](#1-game-overview)
2. [Game Mechanics](#2-game-mechanics)
3. [Reward Design](#3-reward-design)
4. [Observation Space](#4-observation-space)
5. [File Structure](#5-file-structure)
6. [Installation](#6-installation)
7. [Quick Start](#7-quick-start)
8. [Training — Phase 1: MLP + REINFORCE](#8-training--phase-1-mlp--reinforce)
9. [Tracked Metrics](#9-tracked-metrics)
10. [Research Questions](#10-research-questions)

---

## 1. Game Overview

| Property | Value |
|----------|-------|
| Players  | N = 6 (default) |
| Doors    | D = 4 |
| Payout threshold θ | 0.20 |
| History window L | 50 rounds |
| Game type | **Zero-sum**, repeated, partially observable |

Each round consists of one Host and N Players. The Host secretly observes the correct door and emits a **public signal** (which may or may not be truthful). Players observe the signal and simultaneously pick a door. Rewards are distributed based on a dynamic payout multiplier.

---

## 2. Game Mechanics

### Round Sequence

```
1. Environment draws correct_door  (hidden from Players)
2. Host observes correct_door → outputs public_signal ∈ {0,1,2,3}
3. Players observe public_signal → each picks a door ∈ {0,1,2,3}
4. Settlement: compute win-ratio x, apply M(x), distribute rewards
```

### Action Spaces

| Agent | Action | Range |
|-------|--------|-------|
| Host | Which door to signal | `{0, 1, 2, 3}` |
| Player | Which door to pick | `{0, 1, 2, 3}` |

All agents have **fixed bet size = 1 unit** per round (no variable betting).

---

## 3. Reward Design

The reward system is a **dynamic odds game**. The more players win, the lower each winner's payout — discouraging blind herding.

### Definitions

| Symbol | Definition |
|--------|-----------|
| N | Total number of players |
| W | Number of players who picked the correct door |
| x = W/N | Win ratio |
| θ = 0.20 | Host break-even threshold |
| M(x) | Dynamic payout multiplier |

### Payout Multiplier

$$M(x) = 1 + \frac{1 - \theta}{x}, \quad x > 0$$

Returns `0` when `x = 0` (no winners).

### Player Reward

$$r_{\text{player}} = \begin{cases} M(x) - 1 = \dfrac{1-\theta}{x} & \text{winner} \\[6pt] -1 & \text{loser} \end{cases}$$

### Host Reward

$$r_{\text{host}} = N - W \cdot M(x) = N \cdot (\theta - x)$$

- `x < θ` → Host profits
- `x = θ` → Break-even
- `x > θ` → Host loses

### Zero-Sum Verification

$$\underbrace{W \cdot \frac{1-\theta}{x}}_{\text{winners}} + \underbrace{(N-W)\cdot(-1)}_{\text{losers}} + \underbrace{N(\theta - x)}_{\text{host}} = 0 \checkmark$$

### Reward Table (N = 6, θ = 0.20)

| Winners W | x | Winner reward | Host reward |
|-----------|---|--------------|-------------|
| 0 | 0.00 | — | **+6.0** (max) |
| 1 | 0.17 | **+3.8** | +0.2 |
| 2 | 0.33 | +1.4 | −0.8 |
| 3 | 0.50 | +0.6 | −1.8 |
| 6 | 1.00 | −0.2 | −4.8 |

---

## 4. Observation Space

Both agents receive a **flat vector** containing a sliding history window plus the current-round context. Phase 1 (MLP) and Phase 2 (Transformer) use the **same observation**, making the two architectures directly comparable.

### Player Observation — 455 dims

```
[ history  (L × 8)  ]  50 × 8  = 400  dims
[ attn mask (L)     ]          =  50  dims
[ current context   ]  D+1     =   5  dims
─────────────────────────────────────────
                          total = 455  dims
```

**Per-timestep history features (8):**

| Feature | Description |
|---------|-------------|
| `door_choice / (D-1)` | Normalised door this player picked |
| `public_signal / (D-1)` | Normalised signal from Host |
| `followed_signal` | 1 if player followed the signal |
| `won` | 1 if player won |
| `door_ratio[0..3]` | Fraction of all players on each door |

**Current context (5):** `public_signal` one-hot (4) + `round_frac` (1)

### Host Observation — 255 dims

```
[ history  (L × 4)  ]  50 × 4  = 200  dims
[ attn mask (L)     ]          =  50  dims
[ current context   ]  D+1     =   5  dims
─────────────────────────────────────────
                          total = 255  dims
```

**Per-timestep history features (4):**

| Feature | Description |
|---------|-------------|
| `correct_door / (D-1)` | Normalised true door |
| `public_signal / (D-1)` | Signal Host emitted |
| `signal_honest` | 1 if signal == correct_door |
| `win_ratio` | x = W/N that round |

**Current context (5):** `correct_door` one-hot (4) + `round_frac` (1)

> **Attention mask:** 1 for valid timesteps, 0 for padding (first L rounds). Used by the Transformer in Phase 2.

---

## 5. File Structure

```
b12902145/
├── env/
│   ├── oracle_gambit_env.py   # Core MARL environment (PettingZoo AECEnv)
│   ├── verify_env.py          # 5-group mathematical verification (all pass)
│   └── watch_random.py        # Colorful terminal visualisation
│
├── agents/
│   └── mlp_agent.py           # MLP policy: obs → Categorical(D doors)
│
├── training/
│   └── reinforce_runner.py    # REINFORCE + EMA baseline training loop
│
├── experiments/
│   └── run_mlp_baseline.py    # Phase 1 entry point (CLI)
│
├── checkpoints/               # Saved model weights (.pt)
├── requirements.txt
└── README.md
```

---

## 6. Installation

```bash
# Python 3.10 or 3.11 recommended
pip install -r requirements.txt
```

**requirements.txt:**

```
numpy>=1.26
gymnasium>=1.0
pettingzoo>=1.24
torch>=2.2
tensorboard>=2.16
matplotlib>=3.8
```

---

## 7. Quick Start

### Visualise random agents

```bash
python env/watch_random.py --rounds 20 --players 6 --seed 42
```

### Run environment verification (5/5 tests)

```bash
python env/verify_env.py
```

### Use the environment directly

```python
from env.oracle_gambit_env import OracleGambitEnv

env = OracleGambitEnv(num_players=6, num_doors=4, seed=42)
env.reset()

host_action   = 0.33   # float in [0,1] → mapped to door index
player_action = {"player_0": 0.33, "player_1": 0.67, ...}

rewards = env.step_all(host_action, player_action)
# returns {"host": float, "player_0": float, ..., "player_5": float}
```

---

## 8. Training — Phase 1: MLP + REINFORCE

### Architecture

```
obs (455 or 255 dims)
  └─ Linear(obs_dim, 256) + Tanh
       └─ Linear(256, 128) + Tanh
            └─ Linear(128, 4)  ← logits over D doors
                 └─ Categorical → sample door index
```

- **Host agent** and **Player agent** are separate networks.
- All 6 players **share one network** (parameter sharing → 6× sample efficiency).

### Algorithm: REINFORCE with EMA Baseline

$$\mathcal{L} = -\mathbb{E}\left[(r - b)\cdot\log\pi(a\mid o)\right] - \beta\cdot H[\pi]$$

| Symbol | Value |
|--------|-------|
| Baseline $b$ | EMA of batch rewards, momentum α = 0.99 |
| Entropy coeff $\beta$ | 0.01 |
| Optimiser | Adam, lr = 3×10⁻⁴ |
| Batch size | 128 rounds per update |
| Grad clip | L2 norm ≤ 1.0 |

> **No replay buffer** — REINFORCE is on-policy; each batch is collected fresh and discarded after one update.

### Two-Phase Observation (critical detail)

Players must see the Host's signal **before** choosing a door. The runner handles this as:

```
1. host.act(obs)              → door_h  (= public_signal)
2. env._public_signal = door_h          (expose signal early)
3. player_i.act(env.observe(player_i))  (obs now contains door_h)
4. env.step_all(door_h, player_doors)   (step re-derives same door_h)
```

### Run training

```bash
# Default: 100,000 rounds, 6 players
python experiments/run_mlp_baseline.py

# Quick smoke test
python experiments/run_mlp_baseline.py --rounds 2000

# Custom settings
python experiments/run_mlp_baseline.py \
    --rounds 200000 --players 6 --lr 1e-3 --entropy_coeff 0.05
```

Checkpoints are saved to `checkpoints/mlp_reinforce/` and a training curve PNG is generated on completion.

---

## 9. Tracked Metrics

| Metric | Symbol | Meaning |
|--------|--------|---------|
| Host reward | H | Avg host reward per round (positive = host winning) |
| Player reward | P | Avg player reward per round |
| Win ratio | wr | Avg fraction of players who won (random baseline ≈ 0.25) |
| Signal honesty | hon | Fraction of rounds where signal == correct_door |
| Follow rate | fol | Fraction of players who followed the signal |

---

## 10. Research Questions

1. **Trust equilibrium** — Do players learn to distrust a deceptive host? Does the host adapt by becoming more honest?

2. **Nash Equilibrium convergence** — Theory predicts a mixed-strategy equilibrium at some honesty rate $p^*$ and follow rate $q^*$. Does self-play converge near it?

3. **Architecture comparison (Phase 1 vs Phase 2)** — Can a Transformer (with explicit attention over the history) learn faster or converge to a better equilibrium than the MLP baseline?

4. **Herding and anti-herding** — Does the host learn to exploit players who blindly follow signals, creating a "loyal herd" that keeps $x$ low?
