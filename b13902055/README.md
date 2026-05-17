This Markdown is designed to serve as the official documentation for your GitHub repository. It defines the environment logic, the hierarchical decision process, and the technical specifications for the observations and states.

---

# OracleGambit: Strategic Information Manipulation in Multi-Agent RL

**OracleGambit** is a Multi-Agent Reinforcement Learning (MARL) environment designed to study the evolution of deceptive signaling, bribery, and information asymmetry. In this game, a "Host" (the Oracle) possesses ground-truth information but faces a financial conflict of interest with the "Players," leading to complex strategic dynamics.

---

## 1. Environment Dynamics: The 4-Door Game

The environment consists of **$N$ Players** and **1 Host**.

* **The Setup:** There are **4 doors**, only one of which contains the reward.
* **Initial Funds:** Each player starts with a fixed `INITIAL_BALANCE`.
* **The Host:** The Host knows the correct door from the start. The Host has infinite liquidity but its objective is strictly to maximize its net profit (Total Loser Bets - Total Winner Payouts).
* **The Payout Logic:** * If a player wins, they receive a payout.
* If the total winning bets represent a **Minority (< 20% of total bets)**, the Host secures a profit.
* If the winning bets represent a **Majority (≥ 20% of total bets)**, the Host incurs a loss (payouts exceed the pool of loser bets).
* If a player loses all of its Balance, he can't participate the game last.



---

## 2. Game Sequence (Two-Phase Decision)

Each round follows a strict hierarchical sequence:

### Phase I: The Bribery Stage

1. **Bribery:** Every player simultaneously decides on a `Bribe_Fraction` in **[0, 1]** (a proportion of current balance) to offer the Host.  
   The actual paid bribe is `floor(balance * Bribe_Fraction)` in integer dollars, and can be **0**.
2. **Signaling:** * The Host observes all bribes.
* The Host broadcasts a **Public Signal** (a door number, potentially false).
* The Host sends a **Private Signal** to each player. The accuracy/reliability of this signal is a learned strategy by the Host, influenced by the bribe amount.



### Phase II: The Betting Stage

1. **Action:** Players receive the signals and choose:
* Which **Door** to pick (1-4).
* How much to **Bet** (a portion of their current balance).  
  The actual bet is `floor(balance * Bet_Fraction)` in integer dollars, with a minimum bet of **$1** for alive players that can afford at least $1.


2. **Settlement:** The correct door is revealed. Balances are updated, and the Host calculates its net profit/loss.

---

## 3. Technical Specifications: Obs & States

To accommodate the **Transformer** architecture, all observations are structured as fixed-size buffers using **$-1$/None Padding** for future or empty steps.

If a player is out of the game(lose all of his balance), his state in last game presents by $0$ .

### A. Global Parameters

These constants define the constraints of the environment.

| Parameter | Value | Description |
| --- | --- | --- |
| `NUM_PLAYERS` | 10 (Default) | Total number of participating agents. |
| `NUM_DOORS` | 4 | Possible choices for the reward. |
| `INITIAL_BALANCE` | 1000 | Starting capital for each player. |
| `MINORITY_THRESHOLD` | 0.20 | Ratio of bets below which the Host profits. |
| `HISTORY_WINDOW` | 50 | Number of past rounds stored in history. |

### B. Player Observation Space

Each player maintains a local buffer to capture the **POMDP** (Partially Observable Markov Decision Process) nature of the game.

* **History (Fixed Size $L$):**
* `Past_Choices`: Doors picked in previous rounds.
* `Past_Signals`: Public/Private signals received.
* `Past_Ratios`: The betting distribution across the 4 doors in previous rounds.


* **Current Context:**
* `Current_Balance`: Remaining funds.
* `Step_1_Result`: The current bribe sent and the resulting signals (injected after the bribery phase).



### C. Host State Space

The Host has **Full Observability** of the market dynamics.

* **Global History:** * Choices and balances of all players.
* All signals (Public/Private) sent in the past.


* **Market Status:** * Total bribes received in the current round.
* Host’s cumulative profit/loss.
* The correct door in currunt round.



---

## 4. Reward Shaping

### For Players (The Agents)

$$R_{player} = \text{Payout} - \text{Bet_Lost} - \text{Bribe_Amount}$$


*Players must learn if the "Information Gain" from a bribe justifies its cost.*

### For the Host (The Oracle)

$$R_{host} = \sum (\text{Loser_Bets}) - \sum (\text{Winner_Payouts}) + \sum (\text{Bribes})$$


*The Host must balance "Credibility" (giving true signals to attract bribes) against "Survival" (misleading players to avoid majority payouts).*

---

## 5. Model Architecture: Hierarchical Transformer

The agents utilize a **Hierarchical Transformer Encoder**.

1. **Feature Embedding:** Converts history and signals into high-dimensional vectors.
2. **Self-Attention (`nhead=8`):** Identifies correlations between bribe amounts and Host honesty over time.
3. **Dual-Stage Heads:**
* **Bidding Head:** Outputs the `Bribe_Fraction` in **[0, 1]**.
* **Betting Head:** Conditioned on the signal received, outputs the `Door` and `Bet_Amount`.



---

## 6. Research Objectives

* **Evolution of Trust:** Do players learn to ignore the Host if the Host becomes too predatory?
* **Market Manipulation:** Will the Host learn to favor high-bribing players to create a "loyal minority"?
* **Cross-Play Evaluation:** Testing "Naive Players" (from early training) against "Sophisticated Hosts" (from late training) to measure the exploitability of information.

This section defines the core economic engine of your environment. It ensures that the **Host** acts as a "Market Maker" with a specific risk threshold at **25%**, and that the **Players** face diminishing returns as a choice becomes "crowded."

---

## 7. Payout & Reward Mechanism

The reward system is designed as a **Dynamic Odds Game**. Unlike a fixed-multiplier bet (e.g., 1:2), the payout per winning player is inversely proportional to the total winning volume, creating a "Slippage" effect.

### A. Mathematical Definitions

* **$P$ (Total Pool):** The sum of all bets placed by all players in the current round.
* **$W$ (Winning Volume):** The sum of all bets placed on the **correct** door.
* **$x$ (Winning Ratio):** The percentage of the total pool that won ($x = \frac{W}{P}$).
* **$M(x)$ (Payout Multiplier):** The factor by which a winner's bet is multiplied.

### B. The Payout Formula

To satisfy the requirement that the Host breaks even at $x = 20\%$, the individual payout for a winning bet $b_i$ is calculated as:

$$\text{Payout}_i = b_i \times \left( 1 + \frac{0.8}{x} \right)$$

> **Why this formula?**
> * **Inverse Proportionality:** As the winning ratio $x$ increases, the multiplier $\frac{0.75}{x}$ decreases.
> * **Threshold Dynamics:** At exactly $x = 0.2$ (20%), the total payout becomes $W \times (1 + \frac{0.8}{0.2}) = W \times 5$. Since $W$ is $20\%$ of $P$, $5 \times 0.2P = P$. The Host pays out exactly what was collected (Break-even).
> 
> 

---

### C. Host Net Profit/Loss

The Host’s financial outcome per round ($R_{host}$) is determined by the total betting pool minus the total payouts:

$$R_{host} = P - \sum (\text{Payouts}) + \sum (\text{Bribes})$$

#### Strategic Scenarios:

| Winner Ratio ($x$) | Payout Multiplier | Host Outcome | Scenario Description |
| --- | --- | --- | --- |
| **10% (Minority)** | **9x** | **Profit (+0.1P)** | High reward for winners; Host collects surplus from losers. |
| **20% (Threshold)** | **5.0x** | **Break-even (0)** | Total payouts equal the total betting pool. |
| **50% (Majority)** | **2.6x** | **Loss (-0.3P)** | Winners receive less; Host must pay the difference from their own pocket. |

---

### D. Observation Mapping for Transformer

To allow the Agent's **Transformer** to detect these patterns, the following data points are injected into the history buffer:

1. **Relative Crowding:** The ratio of betting volume on each door from the previous 5 steps.
2. **Host Payout Pressure:** A scalar representing how much the Host lost or gained in the previous round.
3. **Bribe-to-Signal Correlation:** A feature mapping the `Bribe_Amount` to the `Multiplier` received, allowing the agent to learn if the Host is "selling" entry into a minority or majority group.

### E. Implementation Parameters

Add these to your `parameters` section for the environment:

```python
# Payout Constants
PAYOUT_THRESHOLD = 0.20  # The break-even point (20%)
SURPLUS_COEFFICIENT = 1 - PAYOUT_THRESHOLD  # 0.8

def calculate_payout(individual_bet, total_winning_vol, total_pool):
    x = total_winning_vol / total_pool
    multiplier = 1 + (SURPLUS_COEFFICIENT / x)
    return individual_bet * multiplier

```

---

## 8. Expected Behavioral Evolutions

1. **Host Manipulation:** The Host will learn that providing **too much** accurate information leads to $x > 20\%$, causing a loss. To survive, the Host must "distribute" false signals or lead different players to different doors to keep $x$ near or below $20\%$.
2. **Player Skepticism:** Players will observe that if "Everyone seems to be betting on Door 1," the payout multiplier will crash ($x$ increases), and the Host's incentive to lie increases. The Transformer should attend to "Betting Ratios" as a signal of **Host Deception Risk**.
