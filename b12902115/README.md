This Markdown is designed to serve as the official documentation for your GitHub repository. It defines the environment logic, the hierarchical decision process, and the technical specifications for the observations and states.

---

# OracleGambit: Strategic Information Manipulation in Multi-Agent RL

**OracleGambit** is a Multi-Agent Reinforcement Learning (MARL) environment designed to study the evolution of deceptive signaling, bribery, and information asymmetry. In this game, a "Host" (the Oracle) possesses ground-truth information but faces a financial conflict of interest with the "Players," leading to complex strategic dynamics.

---

## 1. Environment Dynamics: The Multi-Door Game

The environment consists of **$N$ Players** and **1 Host**.

* **The Setup:** There are **`NUM_DOORS`** possible outcomes (default **4**), only one of which contains the reward. Door count is configurable via `OracleGambitConfig.num_doors`.
* **Initial Funds:** Each player starts with a fixed `INITIAL_BALANCE`.
* **The Host:** The Host knows the correct door from the start. The Host has infinite liquidity but its objective is strictly to maximize its net profit (Total Loser Bets − Total Winner Payouts + Bribes).
* **The Payout Logic:**
  * If a player wins, they receive a payout based on the minority-style multiplier (see §7).
  * If the total winning bets represent a **Minority (< 20% of total bets)**, the Host secures a profit.
  * If the winning bets represent a **Majority (≥ 20% of total bets)**, the Host incurs a loss (payouts exceed the pool of loser bets).
  * If a player loses all of its balance, they cannot participate in subsequent rounds (observation rows are zeroed; `alive = 0`).

---

## 2. Game Sequence (Three Phases per Round)

Each round follows a strict hierarchical sequence:

### Phase I: Bribery (`Phase.BRIBE`)

1. **Bribery:** Every alive player submits a **bribe fraction** in $[0, 1]$ of balance.
   * **Fraction $= 0$:** pay **\$0** (no private channel this round).
   * **Fraction $> 0$** and balance $\geq$ `min_bribe_dollars` (default **\$1**): pay at least **\$1**, capped by balance (integer dollars).
2. The Host observation is updated with all bribe amounts (players do not yet see signals).

### Phase II: Signaling (`Phase.SIGNAL`)

1. The Host observes all bribes and the true winning door.
2. The Host broadcasts a **Public Signal** to everyone (door index $0 \ldots D-1$, potentially false).
3. **Private signals are only delivered to players with `bribe > 0`.** Non-bribers keep private index **$-1$** (no insider hint). Host may still *choose* per-player private actions in training, but the environment only stores them for bribers.

Players still do **not** see raw door numbers in their observation—only abstract signal features described in §3.

### Phase III: Betting (`Phase.BET`)

1. **Player action (belief + bet):** Each player chooses:
   * **Belief mode** (discrete, 3-way)—the environment maps this to a concrete door internally:

   | Value | `PlayerBelief` | Mapped door |
   | ---: | --- | --- |
   | `0` | `BELIEVE_PUBLIC` | Door indicated by the public signal |
   | `1` | `BELIEVE_PRIVATE` | Door from this player's private signal (**only if `bribe > 0`**; otherwise coerced to `RANDOM`) |
   | `2` | `RANDOM` | Uniform random door in $\{0,\ldots,D-1\}$ |

   * **Bet fraction** in $[0, 1]$ of remaining balance (floored to integer dollars; minimum \$1 for alive players who can afford it).

2. **Settlement:** The true winning door is compared to mapped choices; balances and Host profit are updated; history buffers are written.

**API:** `step_bribe` → `step_signal` → `step_bet`, or unified `env.step(action)` keyed by `phase`.

---

## 3. Technical Specifications: Observations & States

To accommodate the **Transformer** architecture, observations are structured dicts with fixed-size history windows. **$-1$** padding marks missing or not-applicable fields.

Eliminated players (`balance ≤ 0`) receive all-zero **`current`** rows (`alive = 0`). **`history` is preserved** from buffer (with `-1` on bribe-gated fields when no bribe) so past rounds remain learnable; only live play stops.

### A. Global Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `num_players` | 10 | Number of player agents. |
| `num_doors` | 4 | Number of doors / signal values. |
| `initial_balance` | 1000 | Starting capital per player. |
| `payout_threshold` | 0.10 | Break-even winning bet share for the Host. |
| `history_window` | 50 | Past rounds stored per buffer. |

Configure via `OracleGambitConfig` in `env.py`.

### B. Player Observation Space

Players operate in a **POMDP**: they never receive the true winning door or other players' private signals. They also **do not** see door indices in their observation—only bribe-conditional signal summaries and aggregate market statistics.

**Structured layout:**

```text
players["current"]  : (num_players, 4)
players["history"]  : (num_players, history_window, 12)
```

Flat vector for training (see `obs_encoding.py`): **`STATE_DIM = 604`** = `4 + 50 × 12`.

#### Current context (4 features per player)

| Index | Field | Description |
| ---: | --- | --- |
| 0 | `alive` | `1` if balance > 0, else `0`. |
| 1 | `balance` | Normalized balance if `normalize_balance_in_obs`. |
| 2 | `bribe_sent` | Bribe paid this round (after `step_bribe`). |
| 3 | `signals_agree` | **Only if `bribe_sent > 0` and signals exist:** `1` if public == private for this player, `0` if they differ. **`-1`** if no bribe or before `Phase.SIGNAL`. |

#### History (12 features per time step per player)

| Index | Field | Description |
| ---: | --- | --- |
| 0 | `my_belief` | This player's belief action: `0` pub / `1` priv / `2` random (`-1` padded). |
| 1 | `signals_agree` | **Only if that round's `bribe > 0`:** public == private (`-1` if no bribe). |
| 2 | `host_public_honest` | `1` if Host's public signal equaled the winning door that round, else `0`. |
| 3 | `bribe_private_hit` | **Only if `bribe > 0`:** `1` if private signal hit the winner, `0` if not. **`-1`** if no bribe. |
| 4 | `bribe` | Bribe amount that round. |
| 5 | `bet` | Bet amount that round. |
| 6 | `reward` | Player reward that round. |
| 7 | `host_profit` | Host profit that round (shared scalar). |
| 8 | `frac_correct` | Share of active bettors whose **mapped door** was correct. |
| 9 | `frac_believe_public` | Share who chose `BELIEVE_PUBLIC`. |
| 10 | `frac_believe_private` | Share who chose `BELIEVE_PRIVATE`. |
| 11 | `frac_random` | Share who chose `RANDOM`. |

> **Design note:** Per-door betting ratios are **not** exposed to players. Market crowding is summarized by belief-mode fractions and `frac_correct`, encouraging learning from collective behavior without indexing doors.

#### Player action spaces

| Phase | Space | Shape / keys |
| --- | --- | --- |
| BRIBE | `player_bribe_action_space` | `(num_players,)` fractions |
| BET | `player_bet_action_space` | `beliefs`: `MultiDiscrete([3] × N)`, `bet_fractions`: `(N,)` |

### C. Host Observation Space

The Host has **full observability** of ground truth and per-player bribes.

```text
host["current"]  : (2 + num_doors,)   # cumulative profit, total bribes, winning door one-hot
host["players"]  : (num_players, 4)   # balance, active, bribe, private_honest_now (-1/0/1)
host["history"]  : (history_window, 7 + num_doors)
host["private_honesty_hist"] : (num_players, history_window)  # past private: 1 honest, 0 lied, -1 no bribe
```

#### Host history (per round)

| Fields | Description |
| --- | --- |
| `host_profit` | Host net profit that round. |
| `public_signal` | Door index broadcast (Host knows what it sent). |
| `frac_correct` | Fraction of bettors on the winning mapped door. |
| `frac_believe_public` / `frac_believe_private` / `frac_random` | Belief-mode shares among bettors. |
| `host_public_honest` | Whether public signal matched the winner. |
| `private_signal_distribution` | Histogram of private signals sent (`num_doors` bins). |

**`private_honesty_hist` (Host only):** For each player $i$ and past round $t$: `1` if they bribed and private matched the winner, `0` if bribed and lied, `-1` if no bribe. Same data as players' `bribe_private_hit` history, exposed to the Host for per-player targeting.

#### Host action space (`Phase.SIGNAL`)

```python
{
    "public_signal": Discrete(num_doors),
    "private_signals": MultiDiscrete([num_doors] * num_players),
}
```

---

## 4. Rewards

### 4.1 Environment rewards (`env.py`)

These are the **true game payoffs** returned by `step()` at settlement (after belief → door mapping).

**Players** (per player $i$ each round):

$$R_{player}^{env} = \text{Payout}_i - b_i - \text{Bribe}_i$$

where $b_i$ is the bet debited from balance and $\text{Bribe}_i$ was already paid in the bribe phase. Losers receive $0$ payout; winners receive $\lfloor b_i \cdot M(x)\rfloor$ (see §7).

**Host** (scalar per round):

$$R_{host}^{env} = P - \sum_i \text{Payout}_i + \sum_i \text{Bribe}_i$$

with $P = \sum_i b_i$ (total pool). This is the net cash flow: all bets enter the pool, winners are paid out, and bribes are kept in full.

Players do **not** see door indices in observations; they infer trust from `bribe_private_hit`, `signals_agree` (only meaningful when `bribe > 0`), and belief-mode fractions in history—not from one-hot door labels.

The Host’s break-even winning **volume share** is **$x = 0.10$** (`payout_threshold = 0.10`). Below 10% on the winning door, the Host tends to profit; above, tends to lose (see §7).

### 4.2 Training reward shaping (`train_both.py`, `train_player.py`)

SAC/RDQN training uses **extra shaping** on top of env rewards (not returned by the environment).

**Player — two-step SAC**

| Step | Symbol | Definition |
| --- | --- | --- |
| Bribe | $r_1$ | $-\text{Bribe}_i + 1.25 \times \mathbb{1}[\text{trusted private last round}] \times \text{Bribe}_i$ |
| Bet | $r_2$ | $R_{player}^{shaped} - r_1$ |

$R_{player}^{shaped}$ starts from $R_{player}^{env}$ and adds:

* $+0.25$ if belief = private and private signal was truthful  
* $+0.20$ if belief = private and player was “trust-profitable” last round  
* $+0.10 \times (1 - \text{crowding})$ on the mapped door  
* $-0.12$ if belief = private and private signal lied  
* $-0.20 \times \text{crowding}$ (share of pool on the player’s mapped door)

**Host — RDQN** (per round, stored in replay):

$$R_{host}^{train} = R_{host}^{env} + 2.0\sum_i \text{Bribe}_i + 0.30 \cdot \overline{\mathbb{1}[\text{priv truthful}]} + 1.20 \cdot \text{weighted truth} - 0.20 \cdot \text{weighted lie}$$

where “weighted truth/lie” weights each player’s private honesty by their bribe. Coefficients match `train_both.py` defaults.

Episode logs (**PubTruth**, **PrivTruth**, belief %) describe behavior; **P_Reward** / **H_Reward** in the console are shaped training signals, not raw balances.

---

## 5. Model Architecture: Hierarchical Transformer

Training code (`train_player.py`, `train_both.py`) uses a **Hierarchical Transformer** over the flat `STATE_DIM` vector:

1. **Feature pass-through:** Current (4) + history (12×50) continuous features (no door one-hot).
2. **Transformer encoder:** Temporal attention over the 50-step history.
3. **Dual-stage heads (Player):**
   * **Bribe actor (SAC):** Continuous bribe fraction.
   * **Bet actor (SAC):** **3-way belief** (Gumbel-Softmax) + continuous bet fraction.

Host co-training (`train_both.py`) uses **RDQN** over Host observations with discrete public/private door outputs (Host still operates in door index space).

**Checkpoint compatibility:** Models trained before the belief/abstract-obs refactor use different action and observation dimensions and are **not** compatible with the current `env.py`.

---

## 6. Research Objectives

* **Evolution of Trust:** Do players learn to ignore signals when `host_public_honest` is low or `bribe_private_hit` is poor?
* **Market Manipulation:** Will the Host favor high-bribing players while keeping `frac_correct` below the payout threshold?
* **Cross-Play Evaluation:** Naive vs. sophisticated policies (`eval.py`, `random_policy.py`) with Rich logs showing belief labels and resolved doors in `info`.

---

## 7. Payout & Reward Mechanism

The reward system is a **dynamic odds game**: payout per winner is inversely related to the winning volume share $x = W/P$.

### A. Mathematical Definitions

* **$P$ (Total Pool):** Sum of all bets in the round.
* **$W$ (Winning Volume):** Sum of bets on the **correct** door (after belief mapping).
* **$x$ (Winning Ratio):** $x = W / P$.
* **$M(x)$ (Payout Multiplier):** Applied to each winning bet.

### B. The Payout Formula

$$\text{Payout}_i = \left\lfloor b_i \times \left( 1 + \frac{0.9}{x} \right) \right\rfloor$$

(with $x$ floored for stability and optional `max_payout_multiplier` cap)

At $x = 0.1$, total payout equals $P$ (Host break-even). Below 10%, Host profits; above, Host loses.

### C. Host Net Profit/Loss

$$R_{host} = P - \sum (\text{Payouts}) + \sum (\text{Bribes})$$

| Winner Ratio ($x$) | Host Outcome (illustrative) |
| --- | --- |
| **5%** | Strong profit — few winners, high multiplier |
| **10%** | Break-even (total payout ≈ $P$) |
| **30%+** | Loss — many winners dilute the pool |

### D. Implementation (`env.py`)

```python
PAYOUT_THRESHOLD = 0.10
SURPLUS_COEFFICIENT = 1 - PAYOUT_THRESHOLD  # 0.9

# x = total_winning_vol / total_pool
multiplier = 1 + (SURPLUS_COEFFICIENT / max(x, min_winning_ratio_for_payout))
payout = floor(bet * multiplier)
```

---

## 8. Expected Behavioral Evolutions

1. **Host Manipulation:** Accurate public signals that push too many bettors onto the winner (via `BELIEVE_PUBLIC` or correlated privates) raise $x$ above 20% and hurt Host profit. The Host may learn to disagree across public/private channels or target bribers selectively.
2. **Player Skepticism:** With `bribe_private_hit` and bribe-gated `signals_agree`, players can associate paid bribes with private accuracy without observing door numbers. Rising `frac_believe_public` when `host_public_honest` is low should be a exploitable crowd signal.
3. **Belief vs. Random:** The `RANDOM` action provides exploration and skepticism when signals are unreliable; `frac_*` history features summarize herd behavior without door-index leakage.

---

## 9. Quick Reference: Environment Files

| File | Role |
| --- | --- |
| `env.py` | `OracleGambitEnv`, `PlayerBelief`, config, step API |
| `obs_encoding.py` | `STATE_DIM`, `flatten_obs`, `encode_features` |
| `train_player.py` | Player-only SAC training |
| `train_both.py` | Player SAC + Host RDQN co-training (CLI, reward shaping) |
| `random_policy.py` | Random baseline with belief/door logging |
| `eval.py` | Load `checkpoints/player.pth` + `host.pth` |
| `audit_env.py` | Runtime invariant checks |

### Training (co-train)

```bash
python train_both.py --num-doors 4 --seed 42 --total-bet-steps 120000
```

Saves `checkpoints/player.pth` and `checkpoints/host.pth` every 50 episodes.
