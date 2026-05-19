# Multi-Agent Co-Training Architecture Design for OracleGambit

This document provides a comprehensive technical overview of the MARL (Multi-Agent Reinforcement Learning) system implemented in `train_both.py`. The setup pits a team of **Players** (trying to maximize balance via smart betting and bribing) against a single **Host** (trying to maximize profit by strategically broadcasting public and private signals).

---

## 1. Player Training Design (Soft Actor-Critic - SAC)

The Player agents operate in a multi-stage environment and are trained using a modified **Two-Step Soft Actor-Critic (SAC)** algorithm. This approach explicitly models the sequence of dependencies within a single environment round (Bribing $
ightarrow$ Observing Signals $
ightarrow$ Betting).

### A. State Representation & Feature Extraction
A Player's observation is flattened into a 1D vector consisting of current features and a rolling window of past round histories.
* **Current Features (Dim: 5)**:
  - Life status (Alive/Dead)
  - Current financial balance
  - Previous round bribe fraction submitted
  - Received public signal index
  - Received private signal index
* **History Window Features**: A sequence of the past $H=50$ rounds. Each round records features like choice made, public signal, private signal, bribes, bets, individual reward, host profit, and specific door betting ratios.
* **Feature Processing (`TransformerExtractor`)**:
  - Continuous elements are preserved natively.
  - Discrete indicators (e.g., signals and past choice doors) are mapped via **One-Hot Encoding** to avoid treating labels as ordinal variables.
  - A multi-layer Transformer-based block processes the sequential history, which is subsequently fused with the linear processing of current features to produce a high-fidelity 256-dimensional semantic representation.

### B. Dual-Actor Architecture
Because a player acts twice within a single environment phase loop, the policy is split into two distinct specialized Actor heads:
1. **Bribe Actor (Continuous Phase 1)**:
   - Takes the state representation at the start of the round (`Phase.BRIBE`).
   - Standard Gaussian Policy outputs a single value transformed via a Sigmoid function representing the portion of capital to invest as a bribe (`bribe_fraction`).
2. **Bet Actor (Hybrid Phase 2)**:
   - Takes the updated state representation after the host publishes signals (`Phase.BET`).
   - **Door Selection Head**: Outflows discrete probabilities across available choices ($M$ doors). Action sampling is performed differentiably using **Gumbel-Softmax** to enable end-to-end backpropagation.
   - **Bet Fraction Head**: Continuous Gaussian policy yielding a Sigmoid-bound value representing the fraction of capital to stake on the chosen door.

### C. Twin-Q Critic Networks
* Two separate Critics monitor the performance of each action stage:
  - **Critic 1 (Bribe Evaluation)**: Estimates $Q_1(s_1, a_{bribe})$.
  - **Critic 2 (Bet/Door Evaluation)**: Estimates $Q_2(s_2, [a_{door\_onehot}, a_{bet\_fraction}])$.
* Both critics utilize a Twin-Q architecture (taking the minimum of two internal networks) to mitigate overestimation bias.

### D. Multi-Stage Reward & Optimization Flow
* **Reward Shaping**:
  - At Step 1 (`Phase.BRIBE`), players encounter an immediate penalty corresponding to their spent bribe: $r_1 = -\text{bribe}$.
  - At Step 3 (`Phase.BET`), they receive their payout reward from the game engine. Their net stage-2 reward is shaped as $r_2 = \text{total\_reward} - r_1$.
* **Temporal Difference updates are chained cross-step**:
  - **Critic 2 Target**: $y_2 = r_2 + \gamma (1 - d) \cdot [\min(Q_{target1}(s_{1,next}, a_{1,next})) - \alpha \log \pi_1(a_{1,next}|s_{1,next})]$
  - **Critic 1 Target**: $y_1 = r_1 + \gamma \cdot [\min(Q_{target2}(s_2, a_2)) - \alpha \log \pi_2(a_2|s_2)]$

---

## 2. Host Training Design (Recurrent Deep Q-Network - RDQN)

The Host operates as a central manipulator trying to optimize its total revenue (withholding payouts by confusing the collective, while encouraging bribes). It treats the environment as a POMDP and utilizes a **Recurrent Deep Q-Network (RDQN)**.

### A. State Representation
The Host observes the holistic state from an authoritative standpoint:
* **Current State**:
  - One-Hot encoding of the *actual* winning door (ground truth).
  - A vector containing the exact bribe values submitted by all $N=10$ players.
* **History Window**:
  - Sequence tracking the past metrics of winning doors, historical global payouts, collective betting entropy, and host profit margins.

### B. Network Architecture (`HostRDQN`)
To correctly identify patterns in player bribery over time, the Host requires temporal memory:
* **Recurrent Component**: An **LSTM** network processes the sequential history window to derive a temporal feature embedding representing player behavioral traits.
* **Fusion Layer**: The hidden state from the LSTM is concatenated with the processed linear representation of current ground truth winning doors and incoming bribes.
* **Multi-Head Q-Outputs**:
  - **Public Signal Head**: A dense layer outputting Q-values for each possible public signal option (size: `num_doors`).
  - **Private Signal Heads**: A list of $N$ independent linear layers (one per player), each outputting Q-values for the private signal directed to that specific player (size: `num_doors` each).

### C. Action Space & Selection Strategy
* **Composite Action Selection**: The host determines its complete strategic output simultaneously:
  - $a_{pub} = \arg\max(Q_{pub}(s, \cdot))$
  - $a_{priv, i} = \arg\max(Q_{priv, i}(s, \cdot))$ for $i \in \{1, \dots, N\}$.
* **Exploration**: Employs an $\epsilon$-greedy schedule where the host selects purely random actions for all channels with probability $\epsilon$, decaying steadily from $1.0$ down to a minimum of $0.05$.

### D. Reward Optimization & Joint Q-Learning
* **Reward Shaping Boost**: To align behavior with profit extraction, the Host's reward function combines natural environment profit with a multiplier on bribery income:
  $$\mathcal{R}_{host} = \text{Profit}_{env} + 2.0 \times \sum_{i=1}^N \text{Bribe}_i$$
* **Joint Action Evaluation**:
  The Critic evaluates the joint state-action value by summing the individual action heads:
  $$Q_{total}(s, a) = Q_{pub}(s, a_{pub}) + \sum_{i=1}^N Q_{priv, i}(s, a_{priv, i})$$
* **Loss Minimization**: Minimized using Mean Squared Error (MSE) against the calculated Temporal Difference target:
  $$y_{host} = \mathcal{R}_{host} + \gamma \cdot \left[ \max Q_{target, pub}(s', \cdot) + \sum_{i=1}^N \max Q_{target, priv, i}(s', \cdot) \right] \cdot (1 - d)$$

---

## 3. Co-Training & System Integration

* **Interdependent Dynamics**: As the Player network learns to route higher bribes to gain access to accurate private signals, the Host network is simultaneously adapting its signaling policy—discovering when to offer real inside information vs. when to deceive heavy spenders to trigger structural liquidations.
* **Synchronization & Stability**: Both structures leverage target network techniques (`soft updates` via $\tau=0.005$ for SAC, and explicit target cloning for the Host RDQN) alongside independent replay experiences (`TwoStepReplayBuffer` and `HostReplayBuffer`) to decouple highly volatile multi-agent cross-talk dynamics.