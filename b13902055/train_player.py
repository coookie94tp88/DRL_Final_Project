import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import copy
import os

# 載入自訂環境
from env import OracleGambitEnv, OracleGambitConfig, Phase

# ==========================================
# 1. 神經網路架構設計 (Neural Networks)
# ==========================================

def flatten_obs(obs_dict, device):
    """將 current 和 history 攤平成一個 1D 向量"""
    curr = torch.FloatTensor(obs_dict["current"]).to(device)       # (N, 5)
    hist = torch.FloatTensor(obs_dict["history"]).to(device)       # (N, 50, 11)
    hist_flat = hist.view(hist.shape[0], -1)                       # (N, 550)
    return torch.cat([curr, hist_flat], dim=1)                     # (N, 555)

class BribeActor(nn.Module):
    """ Actor 1: 負責決定賄賂比例 (連續動作) """
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)

    def forward(self, state):
        feat = self.net(state)
        mu = self.mu_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), -20, 2) # 防止數值爆炸
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample() # Reparameterization trick
        action = torch.sigmoid(x_t) # 壓縮到 0~1
        # 計算 Log Prob 並修正 Sigmoid 的 Jacobian
        log_prob = dist.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class BetActor(nn.Module):
    """ Actor 2: 負責選門 (離散) 與下注比例 (連續) """
    def __init__(self, state_dim, num_doors=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.door_logits_layer = nn.Linear(256, num_doors)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)

    def sample(self, state):
        feat = self.net(state)
        
        # 1. 處理離散的選門 (Gumbel-Softmax Trick)
        door_logits = self.door_logits_layer(feat)
        door_one_hot = F.gumbel_softmax(door_logits, tau=1.0, hard=True) # 輸出 One-hot
        door_probs = F.softmax(door_logits, dim=-1)
        # Log prob 僅計算被選中那個門的機率
        log_prob_door = torch.sum(torch.log(door_probs + 1e-8) * door_one_hot, dim=-1, keepdim=True)
        door_idx = door_one_hot.argmax(dim=-1) # 給環境用的 index

        # 2. 處理連續的下注比例
        bet_mu = self.bet_mu_layer(feat)
        bet_log_std = torch.clamp(self.bet_log_std_layer(feat), -20, 2)
        bet_dist = Normal(bet_mu, bet_log_std.exp())
        x_t = bet_dist.rsample()
        bet_fraction = torch.sigmoid(x_t)
        log_prob_bet = bet_dist.log_prob(x_t) - torch.log(bet_fraction * (1 - bet_fraction) + 1e-6)

        total_log_prob = log_prob_door + log_prob_bet.sum(dim=-1, keepdim=True)
        return door_idx, door_one_hot, bet_fraction, total_log_prob

class Critic(nn.Module):
    """ 通用的 Q 網路 (Critic) """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 雙 Q 網路解決高估問題 (Double Q-learning)
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

# ==========================================
# 2. Replay Buffer (兩步 MDP 特製版)
# ==========================================
class TwoStepReplayBuffer:
    def __init__(self, max_size=100000, state_dim=555):
        self.s1 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a1 = np.zeros((max_size, 1), dtype=np.float32)
        self.r1 = np.zeros((max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a2_door = np.zeros((max_size, 4), dtype=np.float32) # One-hot
        self.a2_bet = np.zeros((max_size, 1), dtype=np.float32)
        self.r2 = np.zeros((max_size, 1), dtype=np.float32)
        self.s1_next = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, s1, a1, r1, s2, a2_door, a2_bet, r2, s1_next, done):
        batch_size = s1.shape[0] # N 位玩家同時塞入
        for i in range(batch_size):
            idx = (self.ptr + i) % self.max_size
            self.s1[idx] = s1[i]
            self.a1[idx] = a1[i]
            self.r1[idx] = r1[i]
            self.s2[idx] = s2[i]
            self.a2_door[idx] = a2_door[i]
            self.a2_bet[idx] = a2_bet[i]
            self.r2[idx] = r2[i]
            self.s1_next[idx] = s1_next[i]
            self.done[idx] = done[i]
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.s1[ind]).to(device),
            torch.FloatTensor(self.a1[ind]).to(device),
            torch.FloatTensor(self.r1[ind]).to(device),
            torch.FloatTensor(self.s2[ind]).to(device),
            torch.FloatTensor(self.a2_door[ind]).to(device),
            torch.FloatTensor(self.a2_bet[ind]).to(device),
            torch.FloatTensor(self.r2[ind]).to(device),
            torch.FloatTensor(self.s1_next[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

# ==========================================
# 3. 自定義主辦方策略 (Top 3 真話)
# ==========================================
def host_policy(bribes, winning_door, num_players, num_doors):
    # 給予隨機的公共訊號
    public_signal = random.randint(0, num_doors - 1)
    private_signals = np.zeros(num_players, dtype=np.int32)
    
    # 找出賄賂金額前 3 名的 index (如果金額一樣，argsort 會自動處理)
    top_3_indices = np.argsort(bribes)[-3:]
    
    for i in range(num_players):
        if i in top_3_indices and bribes[i] > 0: # 有付錢且是前三名才給真話
            private_signals[i] = winning_door
        else:
            private_signals[i] = random.randint(0, num_doors - 1) # 隨機亂說
            
    return public_signal, private_signals

# ==========================================
# 4. 主訓練迴圈
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 初始化環境與維度
    config = OracleGambitConfig(num_players=10, max_rounds=200, initial_balance=1000)
    env = OracleGambitEnv(config=config)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state_dim = 5 + (50 * 11) # current(5) + history(50*11)
    
    # 初始化 SAC 網路
    actor1 = BribeActor(state_dim).to(device)
    critic1 = Critic(state_dim, action_dim=1).to(device)
    critic1_target = copy.deepcopy(critic1)
    
    actor2 = BetActor(state_dim, config.num_doors).to(device)
    critic2 = Critic(state_dim, action_dim=5).to(device) # action = 4(door) + 1(bet)
    critic2_target = copy.deepcopy(critic2)

    optimizer_a1 = optim.Adam(actor1.parameters(), lr=3e-4)
    optimizer_c1 = optim.Adam(critic1.parameters(), lr=3e-4)
    optimizer_a2 = optim.Adam(actor2.parameters(), lr=3e-4)
    optimizer_c2 = optim.Adam(critic2.parameters(), lr=3e-4)
    
    buffer = TwoStepReplayBuffer(max_size=50000, state_dim=state_dim)
    
    gamma = 0.99
    tau = 0.005
    alpha = 0.2 # 熵係數 (可調整探索程度)
    batch_size = 256
    
    obs, info = env.reset()
    s1_dict = obs["players"]
    
    episode_reward = 0
    episodes = 0
    
    print("開始訓練 Two-Step SAC...")
    for step in range(500000): # 總訓練步數
        
        # === 階段 1: 賄賂 (Actor 1) ===
        s1_tensor = flatten_obs(s1_dict, device)
        with torch.no_grad():
            bribe_frac, _ = actor1.sample(s1_tensor)
        
        bribe_action_np = bribe_frac.cpu().numpy().flatten()
        env.step_bribe(bribe_action_np)
        
        # 紀錄實際被扣除的賄賂金額 (r1)
        r1_np = -env.current_bribes.copy() 
        
        # === 階段 2: 主辦方發訊號 ===
        winning_door = env.current_winning_door
        pub_sig, priv_sigs = host_policy(env.current_bribes, winning_door, config.num_players, config.num_doors)
        env.step_signal(pub_sig, priv_sigs)
        
        # 獲取包含訊號的狀態 s2
        obs2 = env._get_observations()
        s2_dict = obs2["players"]
        s2_tensor = flatten_obs(s2_dict, device)
        
        # === 階段 3: 下注 (Actor 2) ===
        with torch.no_grad():
            door_idx, door_onehot, bet_frac, _ = actor2.sample(s2_tensor)
            
        door_idx_np = door_idx.cpu().numpy()
        bet_frac_np = bet_frac.cpu().numpy().flatten()
        
        next_obs, rewards, terminated, truncated, info = env.step_bet(door_idx_np, bet_frac_np)
        
        # 紀錄最終利潤，計算 r2 = 總利潤 - r1 (因為環境回傳的 players 是扣除賄賂後的淨利)
        total_rewards_np = rewards["players"]
        r2_np = total_rewards_np - r1_np 
        
        s1_next_dict = next_obs["players"]
        s1_next_tensor = flatten_obs(s1_next_dict, device)
        
        # === 儲存至 Replay Buffer ===
        done_arr = np.full((config.num_players, 1), terminated)
        buffer.add(
            s1_tensor.cpu().numpy(),
            bribe_frac.cpu().numpy(),
            r1_np.reshape(-1, 1),
            s2_tensor.cpu().numpy(),
            door_onehot.cpu().numpy(),
            bet_frac.cpu().numpy(),
            r2_np.reshape(-1, 1),
            s1_next_tensor.cpu().numpy(),
            done_arr
        )
        
        s1_dict = s1_next_dict
        episode_reward += np.mean(total_rewards_np)
        
        if terminated or truncated:
            obs, info = env.reset()
            s1_dict = obs["players"]
            episodes += 1
            
            # --- 列印訓練進度 ---
            if episodes % 10 == 0:
                print(f"Episode: {episodes}, Avg Reward: {episode_reward/10:.2f}")
            
            # --- 新增：定期保存模型 (例如每 100 局存一次) ---
            if episodes % 100 == 0:
                save_path = os.path.join(checkpoint_dir, f"sac_checkpoint_ep_{episodes}.pth")
                # 我們把 4 個網路的權重包成一個字典存起來
                torch.save({
                    'episode': episodes,
                    'actor1_state_dict': actor1.state_dict(),
                    'actor2_state_dict': actor2.state_dict(),
                    'critic1_state_dict': critic1.state_dict(),
                    'critic2_state_dict': critic2.state_dict(),
                }, save_path)
                print(f"💾 [Checkpoint] 模型已儲存至: {save_path}")

            episode_reward = 0

        # === 神經網路訓練更新 ===
        if buffer.size > batch_size * 2:
            s1, a1, r1, s2, a2_door, a2_bet, r2, s1_next, done = buffer.sample(batch_size, device)
            a2 = torch.cat([a2_door, a2_bet], dim=-1) # (B, 5)

            # ---------------------------------
            # 更新 Critic 2 (下注階段)
            # ---------------------------------
            with torch.no_grad():
                # 取得下一步 (新的 s1) 的價值
                next_a1, next_log_pi1 = actor1.sample(s1_next)
                target_q1_a, target_q1_b = critic1_target(s1_next, next_a1)
                target_v1 = torch.min(target_q1_a, target_q1_b) - alpha * next_log_pi1
                # Q2 的 Target 是 r2 + gamma * (1-done) * V1
                target_q2 = r2 + gamma * (1 - done) * target_v1

            current_q2_a, current_q2_b = critic2(s2, a2)
            q2_loss = F.mse_loss(current_q2_a, target_q2) + F.mse_loss(current_q2_b, target_q2)

            optimizer_c2.zero_grad()
            q2_loss.backward()
            optimizer_c2.step()

            # ---------------------------------
            # 更新 Critic 1 (賄賂階段)
            # ---------------------------------
            with torch.no_grad():
                # 取得下半部 (s2) 的價值
                next_door_idx, next_door_onehot, next_bet_frac, next_log_pi2 = actor2.sample(s2)
                next_a2 = torch.cat([next_door_onehot, next_bet_frac], dim=-1)
                target_q2_a, target_q2_b = critic2_target(s2, next_a2)
                target_v2 = torch.min(target_q2_a, target_q2_b) - alpha * next_log_pi2
                # Q1 的 Target 是 r1 + gamma * V2 (同一回合內一定會走到下注，所以沒有 1-done)
                target_q1 = r1 + gamma * target_v2

            current_q1_a, current_q1_b = critic1(s1, a1)
            q1_loss = F.mse_loss(current_q1_a, target_q1) + F.mse_loss(current_q1_b, target_q1)

            optimizer_c1.zero_grad()
            q1_loss.backward()
            optimizer_c1.step()

            # ---------------------------------
            # 更新 Actor 1 & 2
            # ---------------------------------
            # 更新 Actor 2
            curr_door_idx, curr_door_onehot, curr_bet_frac, log_pi2 = actor2.sample(s2)
            curr_a2 = torch.cat([curr_door_onehot, curr_bet_frac], dim=-1)
            q2_pi_a, q2_pi_b = critic2(s2, curr_a2)
            q2_pi = torch.min(q2_pi_a, q2_pi_b)
            a2_loss = (alpha * log_pi2 - q2_pi).mean()

            optimizer_a2.zero_grad()
            a2_loss.backward()
            optimizer_a2.step()

            # 更新 Actor 1
            curr_a1, log_pi1 = actor1.sample(s1)
            q1_pi_a, q1_pi_b = critic1(s1, curr_a1)
            q1_pi = torch.min(q1_pi_a, q1_pi_b)
            a1_loss = (alpha * log_pi1 - q1_pi).mean()

            optimizer_a1.zero_grad()
            a1_loss.backward()
            optimizer_a1.step()

            # ---------------------------------
            # 更新 Target Networks (Soft Update)
            # ---------------------------------
            for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    final_path = os.path.join(checkpoint_dir, "sac_final_model.pth")
    torch.save({
        'episode': episodes,
        'actor1_state_dict': actor1.state_dict(),
        'actor2_state_dict': actor2.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
    }, final_path)
    print(f"🎉 訓練完成！最終模型已儲存至: {final_path}")
if __name__ == "__main__":
    train()