import argparse
import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from env import OracleGambitConfig, OracleGambitEnv, Phase, PlayerBelief
from obs_encoding import (
    CURR_DIM,
    HIST_DIM,
    SEQ_LEN,
    STATE_DIM,
    encode_features,
    flatten_obs as flatten_obs_player,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


TARGET_Q_CLAMP = 2000.0
REWARD_CLAMP = 500.0


def sanitize_state(state_flat: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(state_flat, nan=0.0, posinf=1.0, neginf=-1.0)


def flatten_obs_safe(obs_dict, device="cpu") -> torch.Tensor:
    return sanitize_state(flatten_obs_player(obs_dict, device))


def clamp_rewards_np(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=REWARD_CLAMP, neginf=-REWARD_CLAMP), -REWARD_CLAMP, REWARD_CLAMP)


def module_has_nan(module: nn.Module) -> bool:
    return any(not torch.isfinite(p).all() for p in module.parameters())


def reinit_linear_modules(*modules: nn.Module) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)


def recover_nan_player_networks(
    actor1, actor2, critic1, critic2, critic1_target, critic2_target
) -> None:
    """Reinitialize all player SAC nets if any weight is non-finite."""
    reinit_linear_modules(actor1, actor2, critic1, critic2)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    print("⚠️  Reset player SAC networks (actors + critics) due to NaN/Inf.")


def safe_opt_step(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    parameters,
    grad_clip_norm: float,
) -> bool:
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True


class TransformerExtractor(nn.Module):
    def __init__(self, curr_dim=CURR_DIM, hist_dim=HIST_DIM, d_model=128, nhead=4, num_layers=2, seq_len=SEQ_LEN):
        super().__init__()
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.curr_proj = nn.Linear(curr_dim, d_model)
        self.fc_out = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU())

    def forward(self, curr_enc, hist_enc):
        x = self.hist_proj(hist_enc) + self.pos_emb
        x = self.transformer(x)
        hist_summary = x[:, -1, :]
        curr_summary = torch.relu(self.curr_proj(curr_enc))
        return self.fc_out(torch.cat([hist_summary, curr_summary], dim=-1))


class BribeActor(nn.Module):
    def __init__(self, log_std_min=-5.0, log_std_max=1.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.extractor = TransformerExtractor()
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        mu = self.mu_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state_flat):
        state_flat = sanitize_state(state_flat)
        mu, log_std = self.forward(state_flat)
        if not torch.isfinite(mu).all():
            mu = torch.zeros_like(mu)
        if not torch.isfinite(log_std).all():
            log_std = torch.zeros_like(log_std)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.sigmoid(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class BetActor(nn.Module):
    NUM_BELIEFS = 3

    def __init__(self, gumbel_tau=0.8, log_std_min=-5.0, log_std_max=1.0):
        super().__init__()
        self.gumbel_tau = gumbel_tau
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.extractor = TransformerExtractor()
        self.belief_logits_layer = nn.Linear(256, self.NUM_BELIEFS)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)

    def sample(self, state_flat):
        state_flat = sanitize_state(state_flat)
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        if not torch.isfinite(feat).all():
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        belief_logits = self.belief_logits_layer(feat)
        belief_one_hot = F.gumbel_softmax(belief_logits, tau=self.gumbel_tau, hard=True)
        belief_probs = F.softmax(belief_logits, dim=-1)
        log_prob_belief = torch.sum(
            torch.log(belief_probs + 1e-8) * belief_one_hot, dim=-1, keepdim=True
        )
        belief_idx = belief_one_hot.argmax(dim=-1)

        bet_mu = self.bet_mu_layer(feat)
        bet_log_std = torch.clamp(self.bet_log_std_layer(feat), self.log_std_min, self.log_std_max)
        bet_dist = Normal(bet_mu, bet_log_std.exp())
        x_t = bet_dist.rsample()
        bet_fraction = torch.sigmoid(x_t)
        log_prob_bet = bet_dist.log_prob(x_t) - torch.log(bet_fraction * (1 - bet_fraction) + 1e-6)

        total_log_prob = log_prob_belief + log_prob_bet.sum(dim=-1, keepdim=True)
        return belief_idx, belief_one_hot, bet_fraction, total_log_prob


class Critic(nn.Module):
    def __init__(self, action_dim=1):
        super().__init__()
        self.ext1 = TransformerExtractor()
        self.q1_head = nn.Sequential(nn.Linear(256 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.ext2 = TransformerExtractor()
        self.q2_head = nn.Sequential(nn.Linear(256 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state_flat, action):
        curr_enc, hist_enc = encode_features(state_flat)
        feat1 = self.ext1(curr_enc, hist_enc)
        q1 = self.q1_head(torch.cat([feat1, action], dim=-1))
        feat2 = self.ext2(curr_enc, hist_enc)
        q2 = self.q2_head(torch.cat([feat2, action], dim=-1))
        return q1, q2


class TwoStepReplayBuffer:
    def __init__(self, max_size=100000, state_dim=STATE_DIM):
        self.s1 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a1 = np.zeros((max_size, 1), dtype=np.float32)
        self.r1 = np.zeros((max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a2_belief = np.zeros((max_size, 3), dtype=np.float32)
        self.a2_bet = np.zeros((max_size, 1), dtype=np.float32)
        self.r2 = np.zeros((max_size, 1), dtype=np.float32)
        self.s1_next = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, s1, a1, r1, s2, a2_belief, a2_bet, r2, s1_next, done):
        batch_size = s1.shape[0]
        for i in range(batch_size):
            idx = (self.ptr + i) % self.max_size
            self.s1[idx] = s1[i]
            self.a1[idx] = a1[i]
            self.r1[idx] = r1[i]
            self.s2[idx] = s2[i]
            self.a2_belief[idx] = a2_belief[i]
            self.a2_bet[idx] = a2_bet[i]
            self.r2[idx] = r2[i]
            self.s1_next[idx] = s1_next[i]
            self.done[idx] = done[i]
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.s1[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.a1[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.r1[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.s2[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.a2_belief[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.a2_bet[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.r2[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.s1_next[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.done[ind], dtype=torch.float32, device=device),
        )


class HostRDQN(nn.Module):
    def __init__(self, num_players=10, num_doors=4, hist_dim=11, d_model=128):
        super().__init__()
        self.num_players = num_players
        self.num_doors = num_doors
        self.lstm = nn.LSTM(input_size=hist_dim, hidden_size=d_model, batch_first=True)
        self.fc_curr = nn.Linear(num_doors + num_players, d_model)
        self.fc_fusion = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU())
        self.q_pub = nn.Linear(256, num_doors)
        self.q_privs = nn.ModuleList([nn.Linear(256, num_doors) for _ in range(num_players)])

    def forward(self, curr, hist):
        _, (h_n, _) = self.lstm(hist)
        hist_feat = h_n[-1]
        curr_feat = F.relu(self.fc_curr(curr))
        merged = self.fc_fusion(torch.cat([hist_feat, curr_feat], dim=-1))
        return self.q_pub(merged), [head(merged) for head in self.q_privs]


def process_host_obs(env, device="cpu"):
    c = env.cfg
    winning_door = env.current_winning_door
    bribes = env.current_bribes
    win_door_oh = np.zeros(c.num_doors, dtype=np.float32)
    if winning_door >= 0:
        win_door_oh[winning_door] = 1.0
    curr_tensor = torch.as_tensor(
        np.concatenate([win_door_oh, bribes]), dtype=torch.float32, device=device
    ).unsqueeze(0)
    hist = env._get_observations()["host"]["history"]
    hist_tensor = torch.as_tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)
    return curr_tensor, hist_tensor


class HostReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, curr_state, hist_state, action, reward, next_curr, next_hist, done):
        self.buffer.append((curr_state, hist_state, action, reward, next_curr, next_hist, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        curr, hist, act, rew, n_curr, n_hist, dn = zip(*batch)
        return (
            torch.cat(curr, dim=0).to(device),
            torch.cat(hist, dim=0).to(device),
            act,
            torch.as_tensor(rew, dtype=torch.float32, device=device).unsqueeze(1),
            torch.cat(n_curr, dim=0).to(device),
            torch.cat(n_hist, dim=0).to(device),
            torch.as_tensor(dn, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


def door_crowding_ratios(chosen_doors: np.ndarray, bets: np.ndarray, num_doors: int, eps: float) -> np.ndarray:
    """Per-player ratio of pool on the door their belief mapped to."""
    pool = float(np.sum(bets))
    totals = np.zeros(num_doors, dtype=np.float32)
    if pool > eps:
        for i, d in enumerate(chosen_doors):
            if bets[i] > 0:
                totals[int(d)] += bets[i]
        totals /= pool
    return totals[np.clip(chosen_doors.astype(int), 0, num_doors - 1)]


def shape_player_rewards(
    base_rewards: np.ndarray,
    beliefs: np.ndarray,
    chosen_doors: np.ndarray,
    bets: np.ndarray,
    private_signals: np.ndarray,
    winning_door: int,
    prev_trust_profit_mask: np.ndarray,
    num_doors: int,
    *,
    truth_follow_bonus: float,
    false_follow_penalty: float,
    trust_profit_follow_bonus: float,
    crowding_penalty: float,
    diversity_bonus: float,
    eps: float,
) -> np.ndarray:
    has_private = (private_signals >= 0).astype(np.float32)
    private_is_truth = ((private_signals == winning_door).astype(np.float32) * has_private)
    follow_private = (
        (beliefs == PlayerBelief.BELIEVE_PRIVATE).astype(np.float32) * has_private
    )
    crowd = door_crowding_ratios(chosen_doors, bets, num_doors, eps)

    shaped = (
        base_rewards
        + truth_follow_bonus * (private_is_truth * follow_private)
        + trust_profit_follow_bonus * (prev_trust_profit_mask * follow_private)
        + diversity_bonus * (1.0 - crowd)
        - false_follow_penalty * ((1.0 - private_is_truth) * follow_private)
        - crowding_penalty * crowd
    )
    return shaped


def shape_host_reward(
    env_host_reward: float,
    raw_bribes: np.ndarray,
    private_signals: np.ndarray,
    winning_door: int,
    *,
    bribe_income_coef: float,
    truth_private_bonus: float,
    weighted_truth_bonus: float,
    weighted_lie_penalty: float,
    eps: float,
) -> float:
    private_is_truth = (private_signals == winning_door).astype(np.float32)
    private_truth_rate = float(np.mean(private_is_truth))
    bribe_weights = np.maximum(raw_bribes, 0.0).astype(np.float32)
    weight_sum = float(np.sum(bribe_weights))
    if weight_sum > eps:
        weighted_truth_rate = float(np.sum(bribe_weights * private_is_truth) / weight_sum)
    else:
        weighted_truth_rate = private_truth_rate
    weighted_lie_rate = 1.0 - weighted_truth_rate
    return (
        env_host_reward
        + bribe_income_coef * float(np.sum(raw_bribes))
        + truth_private_bonus * private_truth_rate
        + weighted_truth_bonus * weighted_truth_rate
        - weighted_lie_penalty * weighted_lie_rate
    )


def save_separate_models(player_actor1, player_actor2, host_rdqn, config, episodes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    player_path = os.path.join(out_dir, "player.pth")
    host_path = os.path.join(out_dir, "host.pth")
    torch.save(
        {
            "episode": episodes,
            "num_doors": config.num_doors,
            "num_players": config.num_players,
            "player_actor1": player_actor1.state_dict(),
            "player_actor2": player_actor2.state_dict(),
        },
        player_path,
    )
    torch.save(
        {
            "episode": episodes,
            "num_doors": config.num_doors,
            "num_players": config.num_players,
            "host_rdqn": host_rdqn.state_dict(),
        },
        host_path,
    )
    return player_path, host_path


def train_both(
    seed=42,
    total_bet_steps=120000,
    batch_size=256,
    save_every_episodes=50,
    num_doors=4,
    player_sac_alpha=0.01,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    config = OracleGambitConfig(
        num_players=10,
        num_doors=num_doors,
        max_rounds=15,
        initial_balance=1000.0,
        history_window=SEQ_LEN,
    )
    env = OracleGambitEnv(config=config, seed=seed)
    state_dim = config.current_player_dim + config.history_window * config.hist_player_dim
    assert state_dim == STATE_DIM, f"STATE_DIM mismatch: {state_dim} vs {STATE_DIM}"

    bribe_log_std_min, bribe_log_std_max = -5.0, 1.0
    bet_log_std_min, bet_log_std_max = -5.0, 1.0
    bet_gumbel_tau = 0.8
    p_gamma1, p_gamma2, p_tau = 0.3, 0.99, 0.005
    grad_clip_norm = 1.0
    phases_per_round_estimate = 4
    max_loop_iters = total_bet_steps * phases_per_round_estimate
    epsilon_stability = 1e-6

    player_truth_follow_bonus = 0.25
    player_false_follow_penalty = 0.12
    player_crowding_penalty = 0.20
    player_diversity_bonus = 0.10
    player_trust_profit_bribe_rebate = 1.25
    player_trust_profit_follow_bonus = 0.20
    host_bribe_income_coef = 2.0
    host_truth_private_bonus = 0.30
    host_weighted_truth_bonus = 1.20
    host_weighted_lie_penalty = 0.20

    player_actor1 = BribeActor(log_std_min=bribe_log_std_min, log_std_max=bribe_log_std_max).to(device)
    player_critic1 = Critic(action_dim=1).to(device)
    player_critic1_target = copy.deepcopy(player_critic1)

    player_actor2 = BetActor(
        gumbel_tau=bet_gumbel_tau,
        log_std_min=bet_log_std_min,
        log_std_max=bet_log_std_max,
    ).to(device)
    player_critic2 = Critic(action_dim=4).to(device)
    player_critic2_target = copy.deepcopy(player_critic2)

    p_opt_a1 = optim.Adam(player_actor1.parameters(), lr=3e-4)
    p_opt_c1 = optim.Adam(player_critic1.parameters(), lr=3e-4)
    p_opt_a2 = optim.Adam(player_actor2.parameters(), lr=3e-4)
    p_opt_c2 = optim.Adam(player_critic2.parameters(), lr=3e-4)

    player_buffer = TwoStepReplayBuffer(max_size=100000, state_dim=state_dim)

    obs, _ = env.reset()
    host_hist_dim = obs["host"]["history"].shape[1]
    host_rdqn = HostRDQN(
        num_players=config.num_players,
        num_doors=config.num_doors,
        hist_dim=host_hist_dim,
    ).to(device)
    host_target_rdqn = copy.deepcopy(host_rdqn)
    h_opt = optim.Adam(host_rdqn.parameters(), lr=1e-3)
    host_buffer = HostReplayBuffer()
    h_gamma, h_epsilon = 0.99, 1.0
    h_eps_decay, h_eps_min = 0.995, 0.05

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    prev_host_curr = prev_host_hist = prev_host_action = None
    last_host_reward = 0.0
    last_private_signals = np.zeros(config.num_players, dtype=np.int32)
    prev_round_trust_profit_mask = np.zeros(config.num_players, dtype=np.float32)
    last_raw_bribes_np = np.zeros(config.num_players, dtype=np.float32)

    episodes = bet_steps = round_in_ep = loop_iters = 0
    ep_player_reward_sum = ep_host_reward_sum = 0.0
    ep_bribe_mean_sum = ep_bet_mean_sum = 0.0
    ep_bribe_steps = ep_bet_steps = ep_signal_steps = 0
    ep_belief_counts = np.zeros(3, dtype=np.float64)
    ep_pub_truth_rate_sum = ep_priv_truth_rate_sum = 0.0
    ep_priv_follow_rate_sum = ep_trust_profit_rate_sum = 0.0

    s2_tensor = None
    r1_np = None
    bribe_frac = None

    print(
        f"🔥 Co-training (belief SAC + host RDQN) | player_sac_alpha={player_sac_alpha} "
        f"(env still uses actor.sample(); lower alpha = less entropy regularization)"
    )
    while bet_steps < total_bet_steps:
        loop_iters += 1
        if loop_iters > max_loop_iters:
            raise RuntimeError(
                f"Training loop exceeded safety cap ({loop_iters} > {max_loop_iters}) "
                f"after {bet_steps} bet steps."
            )

        if env.phase == Phase.BRIBE:
            s1_dict = obs["players"]
            s1_tensor = flatten_obs_safe(s1_dict, device)
            if (
                module_has_nan(player_actor1)
                or module_has_nan(player_actor2)
                or module_has_nan(player_critic1)
                or module_has_nan(player_critic2)
            ):
                recover_nan_player_networks(
                    player_actor1,
                    player_actor2,
                    player_critic1,
                    player_critic2,
                    player_critic1_target,
                    player_critic2_target,
                )
            with torch.no_grad():
                bribe_frac, _ = player_actor1.sample(s1_tensor)
            bribe_action_np = bribe_frac.cpu().numpy().flatten()
            ep_bribe_mean_sum += float(np.mean(bribe_action_np))
            ep_bribe_steps += 1

            obs, _, _, _, _ = env.step({"player_bribe_fractions": bribe_action_np})
            current_bribes_np = env.current_bribes.copy()
            last_raw_bribes_np = current_bribes_np.copy()
            trust_bribe_rebate = (
                player_trust_profit_bribe_rebate * prev_round_trust_profit_mask * current_bribes_np
            )
            r1_np = clamp_rewards_np(-current_bribes_np + trust_bribe_rebate)

        elif env.phase == Phase.SIGNAL:
            curr_tensor, hist_tensor = process_host_obs(env, device=device)
            if prev_host_curr is not None:
                host_buffer.push(
                    prev_host_curr,
                    prev_host_hist,
                    prev_host_action,
                    last_host_reward,
                    curr_tensor,
                    hist_tensor,
                    done=0,
                )

            if random.random() < h_epsilon:
                pub_act = random.randint(0, config.num_doors - 1)
                priv_acts = [random.randint(0, config.num_doors - 1) for _ in range(config.num_players)]
            else:
                with torch.no_grad():
                    q_pub, q_privs = host_rdqn(curr_tensor, hist_tensor)
                    pub_act = q_pub.argmax(dim=-1).item()
                    priv_acts = [q.argmax(dim=-1).item() for q in q_privs]

            host_action = {
                "public_signal": pub_act,
                "private_signals": np.array(priv_acts, dtype=np.int32),
            }
            prev_host_curr = curr_tensor
            prev_host_hist = hist_tensor
            prev_host_action = host_action

            winning_door = env.current_winning_door
            if winning_door >= 0:
                ep_pub_truth_rate_sum += float(pub_act == winning_door)
                bribed_mask = env.current_bribes > 0
                if np.any(bribed_mask):
                    priv_arr = np.array(priv_acts, dtype=np.int32)
                    ep_priv_truth_rate_sum += float(
                        np.mean(priv_arr[bribed_mask] == winning_door)
                    )
                ep_signal_steps += 1

            obs, _, _, _, _ = env.step(host_action)
            last_private_signals = env.current_private_signals.copy()
            s2_tensor = flatten_obs_safe(obs["players"], device)

        elif env.phase == Phase.BET:
            with torch.no_grad():
                belief_idx, belief_onehot, bet_frac_tensor, _ = player_actor2.sample(s2_tensor)

            belief_idx_np = belief_idx.cpu().numpy()
            bet_frac_np = bet_frac_tensor.cpu().numpy().flatten()
            ep_bet_mean_sum += float(np.mean(bet_frac_np))
            ep_bet_steps += 1
            for b in belief_idx_np:
                ep_belief_counts[int(b)] += 1

            next_obs, rewards, terminated, truncated, info = env.step(
                {"player_beliefs": belief_idx_np, "bet_fractions": bet_frac_np}
            )
            bet_steps += 1
            round_in_ep += 1

            if "winning_door" not in info:
                raise KeyError("Expected winning_door in step info during BET phase.")
            winning_door = int(info["winning_door"])
            chosen_doors = info["chosen_doors"]
            bets = env.hist_bets[-1]

            last_host_reward = shape_host_reward(
                float(rewards["host"]),
                last_raw_bribes_np,
                last_private_signals,
                winning_door,
                bribe_income_coef=host_bribe_income_coef,
                truth_private_bonus=host_truth_private_bonus,
                weighted_truth_bonus=host_weighted_truth_bonus,
                weighted_lie_penalty=host_weighted_lie_penalty,
                eps=epsilon_stability,
            )

            follow_private = (belief_idx_np == PlayerBelief.BELIEVE_PRIVATE).astype(np.float32)
            ep_priv_follow_rate_sum += float(np.mean(follow_private))

            shaped_player_rewards = shape_player_rewards(
                rewards["players"],
                belief_idx_np,
                chosen_doors,
                bets,
                last_private_signals,
                winning_door,
                prev_round_trust_profit_mask,
                config.num_doors,
                truth_follow_bonus=player_truth_follow_bonus,
                false_follow_penalty=player_false_follow_penalty,
                trust_profit_follow_bonus=player_trust_profit_follow_bonus,
                crowding_penalty=player_crowding_penalty,
                diversity_bonus=player_diversity_bonus,
                eps=epsilon_stability,
            )
            prev_round_trust_profit_mask = (
                (last_private_signals == winning_door).astype(np.float32)
                * (rewards["players"] > 0.0).astype(np.float32)
            )
            ep_trust_profit_rate_sum += float(np.mean(prev_round_trust_profit_mask))
            r2_np = clamp_rewards_np(shaped_player_rewards - r1_np)

            s1_next_tensor = flatten_obs_safe(next_obs["players"], device)
            done_arr = np.full(
                (config.num_players, 1), float(terminated or truncated), dtype=np.float32
            )
            player_buffer.add(
                s1_tensor.cpu().numpy(),
                bribe_frac.cpu().numpy(),
                r1_np.reshape(-1, 1),
                s2_tensor.cpu().numpy(),
                belief_onehot.cpu().numpy(),
                bet_frac_tensor.cpu().numpy(),
                r2_np.reshape(-1, 1),
                s1_next_tensor.cpu().numpy(),
                done_arr,
            )
            ep_player_reward_sum += float(np.mean(shaped_player_rewards))
            ep_host_reward_sum += last_host_reward
            obs = next_obs

            if player_buffer.size >= batch_size * 2:
                s1, a1, r1, s2, a2_belief, a2_bet, r2, s1_next, done = player_buffer.sample(
                    batch_size, device
                )
                a2 = torch.cat([a2_belief, a2_bet], dim=-1)

                s1 = sanitize_state(s1)
                s2 = sanitize_state(s2)
                s1_next = sanitize_state(s1_next)

                with torch.no_grad():
                    next_a1, next_log_pi1 = player_actor1.sample(s1_next)
                    tq1a, tq1b = player_critic1_target(s1_next, next_a1)
                    target_v1 = torch.min(tq1a, tq1b) - player_sac_alpha * next_log_pi1
                    target_q2 = torch.clamp(
                        r2 + p_gamma1 * (1 - done) * target_v1, -TARGET_Q_CLAMP, TARGET_Q_CLAMP
                    )

                cq2a, cq2b = player_critic2(s2, a2)
                q2_loss = F.mse_loss(cq2a, target_q2) + F.mse_loss(cq2b, target_q2)
                safe_opt_step(q2_loss, p_opt_c2, player_critic2.parameters(), grad_clip_norm)

                with torch.no_grad():
                    _, next_belief_oh, next_bet, next_log_pi2 = player_actor2.sample(s2)
                    next_a2 = torch.cat([next_belief_oh, next_bet], dim=-1)
                    tq2a, tq2b = player_critic2_target(s2, next_a2)
                    target_v2 = torch.min(tq2a, tq2b) - player_sac_alpha * next_log_pi2
                    target_q1 = torch.clamp(
                        r1 + p_gamma2 * target_v2, -TARGET_Q_CLAMP, TARGET_Q_CLAMP
                    )

                cq1a, cq1b = player_critic1(s1, a1)
                q1_loss = F.mse_loss(cq1a, target_q1) + F.mse_loss(cq1b, target_q1)
                safe_opt_step(q1_loss, p_opt_c1, player_critic1.parameters(), grad_clip_norm)

                _, curr_belief_oh, curr_bet, log_pi2 = player_actor2.sample(s2)
                curr_a2 = torch.cat([curr_belief_oh, curr_bet], dim=-1)
                q2_pi_a, q2_pi_b = player_critic2(s2, curr_a2)
                a2_loss = (player_sac_alpha * log_pi2 - torch.min(q2_pi_a, q2_pi_b)).mean()
                safe_opt_step(a2_loss, p_opt_a2, player_actor2.parameters(), grad_clip_norm)

                curr_a1, log_pi1 = player_actor1.sample(s1)
                q1_pi_a, q1_pi_b = player_critic1(s1, curr_a1)
                a1_loss = (player_sac_alpha * log_pi1 - torch.min(q1_pi_a, q1_pi_b)).mean()
                safe_opt_step(a1_loss, p_opt_a1, player_actor1.parameters(), grad_clip_norm)

                for p, tp in zip(player_critic1.parameters(), player_critic1_target.parameters()):
                    tp.data.copy_(p_tau * p.data + (1 - p_tau) * tp.data)
                for p, tp in zip(player_critic2.parameters(), player_critic2_target.parameters()):
                    tp.data.copy_(p_tau * p.data + (1 - p_tau) * tp.data)

            if len(host_buffer) >= batch_size:
                b_curr, b_hist, b_acts, b_rew, b_ncurr, b_nhist, b_done = host_buffer.sample(
                    batch_size, device
                )
                b_act_pub = torch.tensor(
                    [a["public_signal"] for a in b_acts], dtype=torch.long, device=device
                ).view(-1, 1)
                priv_matrix = np.stack([a["private_signals"] for a in b_acts], axis=0)
                b_act_privs = torch.as_tensor(priv_matrix, dtype=torch.long, device=device)

                q_pub, q_privs = host_rdqn(b_curr, b_hist)
                q_pub_val = q_pub.gather(1, b_act_pub)
                q_priv_vals = sum(
                    q_privs[i].gather(1, b_act_privs[:, i].view(-1, 1))
                    for i in range(config.num_players)
                )
                total_q = q_pub_val + q_priv_vals

                with torch.no_grad():
                    next_q_pub, next_q_privs = host_target_rdqn(b_ncurr, b_nhist)
                    max_next_q = next_q_pub.max(1)[0].unsqueeze(1)
                    max_next_q += sum(q.max(1)[0].unsqueeze(1) for q in next_q_privs)
                    target_q = b_rew + h_gamma * max_next_q * (1 - b_done)

                h_loss = F.mse_loss(total_q, target_q)
                h_opt.zero_grad()
                h_loss.backward()
                torch.nn.utils.clip_grad_norm_(host_rdqn.parameters(), grad_clip_norm)
                h_opt.step()

                for p, tp in zip(host_rdqn.parameters(), host_target_rdqn.parameters()):
                    tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

            if terminated or truncated:
                if prev_host_curr is not None:
                    curr_tensor, hist_tensor = process_host_obs(env, device=device)
                    host_buffer.push(
                        prev_host_curr,
                        prev_host_hist,
                        prev_host_action,
                        last_host_reward,
                        curr_tensor,
                        hist_tensor,
                        done=1,
                    )
                prev_host_curr = prev_host_hist = prev_host_action = None
                h_epsilon = max(h_eps_min, h_epsilon * h_eps_decay)
                episodes += 1

                total_beliefs = np.sum(ep_belief_counts) + epsilon_stability
                belief_pct = (ep_belief_counts / total_beliefs) * 100.0
                avg_bribe = ep_bribe_mean_sum / max(1, ep_bribe_steps)
                avg_bet = ep_bet_mean_sum / max(1, ep_bet_steps)
                avg_pub_truth = ep_pub_truth_rate_sum / max(1, ep_signal_steps)
                avg_priv_truth = ep_priv_truth_rate_sum / max(1, ep_signal_steps)
                avg_priv_follow = ep_priv_follow_rate_sum / max(1, ep_bet_steps)
                avg_trust_profit = ep_trust_profit_rate_sum / max(1, ep_bet_steps)
                avg_player_reward = ep_player_reward_sum / max(1, round_in_ep)

                print(
                    f"✅ Ep {episodes:4d} | Rounds {round_in_ep:3d} | "
                    f"P_Reward: {avg_player_reward:+.2f} | H_Reward: {ep_host_reward_sum:+.2f} | "
                    f"H_Eps: {h_epsilon:.3f}"
                )
                print(
                    f"   📊 BribeFrac: {avg_bribe:.3f} | BetFrac: {avg_bet:.3f} | "
                    f"PubTruth: {avg_pub_truth:.3f} | PrivTruth: {avg_priv_truth:.3f} | "
                    f"BeliefPrivFollow: {avg_priv_follow:.3f} | TrustProfit: {avg_trust_profit:.3f}"
                )
                print(
                    f"   🎯 Beliefs: pub {belief_pct[0]:.1f}% | priv {belief_pct[1]:.1f}% | rnd {belief_pct[2]:.1f}%"
                )
                print("-" * 70)

                if episodes % save_every_episodes == 0:
                    player_path, host_path = save_separate_models(
                        player_actor1, player_actor2, host_rdqn, config, episodes, out_dir
                    )
                    print(f"💾 Saved: {player_path} and {host_path}")

                obs, _ = env.reset()
                round_in_ep = 0
                ep_player_reward_sum = ep_host_reward_sum = 0.0
                ep_bribe_mean_sum = ep_bet_mean_sum = 0.0
                ep_bribe_steps = ep_bet_steps = ep_signal_steps = 0
                ep_belief_counts.fill(0.0)
                ep_pub_truth_rate_sum = ep_priv_truth_rate_sum = 0.0
                ep_priv_follow_rate_sum = ep_trust_profit_rate_sum = 0.0
                prev_round_trust_profit_mask.fill(0.0)
                last_private_signals.fill(0)
                last_raw_bribes_np.fill(0.0)

    player_path, host_path = save_separate_models(
        player_actor1, player_actor2, host_rdqn, config, episodes, out_dir
    )
    print(f"✅ Training done.\n - {player_path}\n - {host_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-doors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-bet-steps", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every-episodes", type=int, default=50)
    parser.add_argument(
        "--sac-alpha",
        type=float,
        default=0.01,
        help="SAC entropy coefficient for player actors (0 = off; default 0.01 avoids dominating shaping)",
    )
    args = parser.parse_args()
    train_both(
        seed=args.seed,
        total_bet_steps=args.total_bet_steps,
        batch_size=args.batch_size,
        save_every_episodes=args.save_every_episodes,
        num_doors=args.num_doors,
        player_sac_alpha=args.sac_alpha,
    )
