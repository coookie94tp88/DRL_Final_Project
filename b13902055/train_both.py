import os
import random
import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# 載入自訂環境
from env import OracleGambitEnv, OracleGambitConfig, Phase

# ==========================================
# 1. Player (SAC) 特徵處理與網路架構
# ==========================================

def flatten_obs_player(obs_dict, device):
    """將 Player 的 current 和 history 攤平成一個 1D 向量"""
    curr = torch.FloatTensor(obs_dict["current"]).to(device)       # (N, 5)
    hist = torch.FloatTensor(obs_dict["history"]).to(device)       # (N, 50, 11)
    hist_flat = hist.view(hist.shape[0], -1)                       # (N, 550)
    return torch.cat([curr, hist_flat], dim=1)                     # (N, 555)

def encode_features(state_flat, num_doors=4, seq_len=50):
    """將 555 維資料還原，並對離散訊號進行 One-Hot Encoding"""
    N = state_flat.shape[0]
    curr = state_flat[:, :5]
    hist = state_flat[:, 5:].view(N, seq_len, 11)
    
    # ─── 處理 Current 狀態 ───
    alive_bal_bribe = curr[:, 0:3] 
    pub_sig = curr[:, 3].long()
    priv_sig = curr[:, 4].long()
    
    pub_sig = torch.where(pub_sig < 0, num_doors, pub_sig)
    priv_sig = torch.where(priv_sig < 0, num_doors, priv_sig)
    
    pub_onehot = F.one_hot(pub_sig, num_classes=num_doors+1).float()
    priv_onehot = F.one_hot(priv_sig, num_classes=num_doors+1).float()
    curr_encoded = torch.cat([alive_bal_bribe, pub_onehot, priv_onehot], dim=1)
    
    # ─── 處理 History 狀態 ───
    choice = hist[:, :, 0].long()
    h_pub = hist[:, :, 1].long()
    h_priv = hist[:, :, 2].long()
    continuous_hist = hist[:, :, 3:] 
    
    choice = torch.where(choice < 0, num_doors, choice)
    h_pub = torch.where(h_pub < 0, num_doors, h_pub)
    h_priv = torch.where(h_priv < 0, num_doors, h_priv)
    
    choice_oh = F.one_hot(choice, num_classes=num_doors+1).float()
    h_pub_oh = F.one_hot(h_pub, num_classes=num_doors+1).float()
    h_priv_oh = F.one_hot(h_priv, num_classes=num_doors+1).float()
    
    hist_encoded = torch.cat([choice_oh, h_pub_oh, h_priv_oh, continuous_hist], dim=2)
    return curr_encoded, hist_encoded

class TransformerExtractor(nn.Module):
    def __init__(self, curr_dim=13, hist_dim=23, d_model=128, nhead=4, num_layers=2, seq_len=50):
        super().__init__()
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.curr_proj = nn.Linear(curr_dim, d_model)
        self.fc_out = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU())
        
    def forward(self, curr_enc, hist_enc):
        x = self.hist_proj(hist_enc) + self.pos_emb       
        x = self.transformer(x)                           
        hist_summary = x[:, -1, :]                        
        
        curr_summary = torch.relu(self.curr_proj(curr_enc)) 
        out = self.fc_out(torch.cat([hist_summary, curr_summary], dim=-1)) 
        return out

class BribeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = TransformerExtractor()
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        mu = self.mu_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), -20, 2)
        return mu, log_std

    def sample(self, state_flat):
        mu, log_std = self.forward(state_flat)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.sigmoid(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class BetActor(nn.Module):
    def __init__(self, num_doors=4):
        super().__init__()
        self.extractor = TransformerExtractor()
        self.door_logits_layer = nn.Linear(256, num_doors)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)

    def sample(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        
        door_logits = self.door_logits_layer(feat)
        door_one_hot = F.gumbel_softmax(door_logits, tau=1.0, hard=True)
        door_probs = F.softmax(door_logits, dim=-1)
        log_prob_door = torch.sum(torch.log(door_probs + 1e-8) * door_one_hot, dim=-1, keepdim=True)
        door_idx = door_one_hot.argmax(dim=-1)

        bet_mu = self.bet_mu_layer(feat)
        bet_log_std = torch.clamp(self.bet_log_std_layer(feat), -20, 2)
        bet_dist = Normal(bet_mu, bet_log_std.exp())
        x_t = bet_dist.rsample()
        bet_fraction = torch.sigmoid(x_t)
        log_prob_bet = bet_dist.log_prob(x_t) - torch.log(bet_fraction * (1 - bet_fraction) + 1e-6)

        total_log_prob = log_prob_door + log_prob_bet.sum(dim=-1, keepdim=True)
        return door_idx, door_one_hot, bet_fraction, total_log_prob

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
    def __init__(self, max_size=100000, state_dim=555):
        self.s1 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a1 = np.zeros((max_size, 1), dtype=np.float32)
        self.r1 = np.zeros((max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a2_door = np.zeros((max_size, 4), dtype=np.float32)
        self.a2_bet = np.zeros((max_size, 1), dtype=np.float32)
        self.r2 = np.zeros((max_size, 1), dtype=np.float32)
        self.s1_next = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, s1, a1, r1, s2, a2_door, a2_bet, r2, s1_next, done):
        batch_size = s1.shape[0] 
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
# 2. Host (RDQN) 特徵處理與網路架構
# ==========================================

class HostRDQN(nn.Module):
    def __init__(self, num_players=10, num_doors=4, hist_dim=10, d_model=128):
        super().__init__()
        self.num_players = num_players
        self.num_doors = num_doors
        
        # 歷史序列處理 (LSTM)
        self.lstm = nn.LSTM(input_size=hist_dim, hidden_size=d_model, batch_first=True)
        
        # 當下狀態 (Winning Door [One-Hot: 4] + Bribes [10]) = 14 維
        self.fc_curr = nn.Linear(num_doors + num_players, d_model)
        
        # 融合層
        self.fc_fusion = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU())
        
        # 動作分支 (Q-values)
        self.q_pub = nn.Linear(256, num_doors) 
        self.q_privs = nn.ModuleList([nn.Linear(256, num_doors) for _ in range(num_players)]) 

    def forward(self, curr, hist):
        _, (h_n, _) = self.lstm(hist)
        hist_feat = h_n[-1] # (Batch, d_model)
        
        curr_feat = F.relu(self.fc_curr(curr)) 
        merged = self.fc_fusion(torch.cat([hist_feat, curr_feat], dim=-1))
        
        q_pub_vals = self.q_pub(merged)
        q_priv_vals = [head(merged) for head in self.q_privs]
        return q_pub_vals, q_priv_vals

def process_host_obs(env, device="cpu"):
    """直接從 env 提取 Host 所需特徵，避免複雜的 dict 解析"""
    c = env.cfg
    winning_door = env.current_winning_door
    bribes = env.current_bribes
    
    win_door_oh = np.zeros(c.num_doors, dtype=np.float32)
    if winning_door >= 0:
        win_door_oh[winning_door] = 1.0
        
    curr_processed = np.concatenate([win_door_oh, bribes])
    curr_tensor = torch.FloatTensor(curr_processed).unsqueeze(0).to(device)
    
    # 提取 history
    hist = env._get_observations()["host"]["history"]
    hist_tensor = torch.FloatTensor(hist).unsqueeze(0).to(device)
    
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
            torch.FloatTensor(rew).unsqueeze(1).to(device),
            torch.cat(n_curr, dim=0).to(device), 
            torch.cat(n_hist, dim=0).to(device),
            torch.FloatTensor(dn).unsqueeze(1).to(device)
        )
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. 聯合訓練主迴圈 (Co-Training Loop)
# ==========================================

def train_both():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = OracleGambitConfig(num_players=10, num_doors=4, max_rounds=200)
    env = OracleGambitEnv(config=config, seed=42)
    
    # --- 初始化 Player 網路 (SAC) ---
    player_actor1 = BribeActor().to(device)
    player_critic1 = Critic(action_dim=1).to(device)
    player_critic1_target = copy.deepcopy(player_critic1)
    
    player_actor2 = BetActor(num_doors=config.num_doors).to(device)
    player_critic2 = Critic(action_dim=5).to(device) 
    player_critic2_target = copy.deepcopy(player_critic2)

    p_opt_a1 = optim.Adam(player_actor1.parameters(), lr=3e-4)
    p_opt_c1 = optim.Adam(player_critic1.parameters(), lr=3e-4)
    p_opt_a2 = optim.Adam(player_actor2.parameters(), lr=3e-4)
    p_opt_c2 = optim.Adam(player_critic2.parameters(), lr=3e-4)
    
    player_buffer = TwoStepReplayBuffer(max_size=100000, state_dim=555)
    p_gamma, p_tau, p_alpha = 0.99, 0.005, 0.2
    
    # --- 初始化 Host 網路 (RDQN) ---
    obs, info = env.reset()
    host_hist_dim = obs["host"]["history"].shape[1]
    
    host_rdqn = HostRDQN(num_players=config.num_players, num_doors=config.num_doors, hist_dim=host_hist_dim).to(device)
    host_target_rdqn = copy.deepcopy(host_rdqn)
    h_opt = optim.Adam(host_rdqn.parameters(), lr=1e-3)
    
    host_buffer = HostReplayBuffer()
    h_gamma, h_epsilon = 0.99, 1.0
    h_eps_decay, h_eps_min = 0.995, 0.05
    
    batch_size = 256
    
    # --- 狀態暫存 ---
    prev_host_curr = None
    prev_host_hist = None
    prev_host_action = None
    last_host_reward = 0.0

    episodes = 0
    step = 0
    episode_p_reward = 0
    episode_h_reward = 0

    print("🔥 開始聯合訓練 (Co-Training MARL)...")

    for total_steps in range(1, 1000000): # 無限迴圈或設定一個極大值
        
        # ==========================================
        # Phase 0: 賄賂 (Player Action)
        # ==========================================
        if env.phase == Phase.BRIBE:
            s1_dict = obs["players"]
            s1_tensor = flatten_obs_player(s1_dict, device)
            
            with torch.no_grad():
                bribe_frac, _ = player_actor1.sample(s1_tensor)
            
            bribe_action_np = bribe_frac.cpu().numpy().flatten()
            obs, _, _, _, info = env.step({"player_bribe_fractions": bribe_action_np})
            r1_np = -env.current_bribes.copy() 
            
        # ==========================================
        # Phase 1: 發送訊號 (Host Action)
        # ==========================================
        elif env.phase == Phase.SIGNAL:
            curr_tensor, hist_tensor = process_host_obs(env, device=device)
            
            # 將前一局的結果存入 Buffer
            if prev_host_curr is not None:
                host_buffer.push(
                    prev_host_curr, prev_host_hist, prev_host_action, 
                    last_host_reward, curr_tensor, hist_tensor, done=0
                )

            # Epsilon-Greedy
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
                "private_signals": np.array(priv_acts, dtype=np.int32)
            }
            
            prev_host_curr = curr_tensor
            prev_host_hist = hist_tensor
            prev_host_action = host_action
            
            obs, _, _, _, info = env.step(host_action)

            # 獲取給 Player 用的 s2
            s2_dict = obs["players"]
            s2_tensor = flatten_obs_player(s2_dict, device)

        # ==========================================
        # Phase 2: 下注 (Player Action) & 結算 & 網路更新
        # ==========================================
        elif env.phase == Phase.BET:
            with torch.no_grad():
                door_idx, door_onehot, bet_frac, _ = player_actor2.sample(s2_tensor)
                
            door_idx_np = door_idx.cpu().numpy()
            bet_frac_np = bet_frac.cpu().numpy().flatten()
            
            next_obs, rewards, terminated, truncated, info = env.step({
                "player_doors": door_idx_np,
                "bet_fractions": bet_frac_np
            })
            step += 1
            
            # 處理 Reward 與 Next State
            last_host_reward = rewards["host"]
            total_rewards_np = rewards["players"]
            r2_np = total_rewards_np - r1_np 
            
            s1_next_dict = next_obs["players"]
            s1_next_tensor = flatten_obs_player(s1_next_dict, device)
            done_arr = np.full((config.num_players, 1), terminated)
            
            player_buffer.add(
                s1_tensor.cpu().numpy(), bribe_frac.cpu().numpy(), r1_np.reshape(-1, 1),
                s2_tensor.cpu().numpy(), door_onehot.cpu().numpy(), bet_frac.cpu().numpy(),
                r2_np.reshape(-1, 1), s1_next_tensor.cpu().numpy(), done_arr
            )
            
            episode_p_reward += np.mean(total_rewards_np)
            episode_h_reward += last_host_reward
            obs = next_obs
            
            # ---------------------------------------------
            # 【GPU 火力全開】每步都更新神經網路！
            # ---------------------------------------------
            # 1. 更新 Player SAC
            if player_buffer.size > batch_size * 2:
                s1, a1, r1, s2, a2_door, a2_bet, r2, s1_next, done = player_buffer.sample(batch_size, device)
                a2 = torch.cat([a2_door, a2_bet], dim=-1)

                with torch.no_grad():
                    next_a1, next_log_pi1 = player_actor1.sample(s1_next)
                    target_q1_a, target_q1_b = player_critic1_target(s1_next, next_a1)
                    target_v1 = torch.min(target_q1_a, target_q1_b) - p_alpha * next_log_pi1
                    target_q2 = r2 + p_gamma * (1 - done) * target_v1

                current_q2_a, current_q2_b = player_critic2(s2, a2)
                q2_loss = F.mse_loss(current_q2_a, target_q2) + F.mse_loss(current_q2_b, target_q2)

                p_opt_c2.zero_grad()
                q2_loss.backward()
                p_opt_c2.step()

                with torch.no_grad():
                    _, next_door_onehot, next_bet_frac, next_log_pi2 = player_actor2.sample(s2)
                    next_a2 = torch.cat([next_door_onehot, next_bet_frac], dim=-1)
                    target_q2_a, target_q2_b = player_critic2_target(s2, next_a2)
                    target_v2 = torch.min(target_q2_a, target_q2_b) - p_alpha * next_log_pi2
                    target_q1 = r1 + p_gamma * target_v2

                current_q1_a, current_q1_b = player_critic1(s1, a1)
                q1_loss = F.mse_loss(current_q1_a, target_q1) + F.mse_loss(current_q1_b, target_q1)

                p_opt_c1.zero_grad()
                q1_loss.backward()
                p_opt_c1.step()

                # Update Actors
                _, curr_door_onehot, curr_bet_frac, log_pi2 = player_actor2.sample(s2)
                curr_a2 = torch.cat([curr_door_onehot, curr_bet_frac], dim=-1)
                q2_pi_a, q2_pi_b = player_critic2(s2, curr_a2)
                a2_loss = (p_alpha * log_pi2 - torch.min(q2_pi_a, q2_pi_b)).mean()
                p_opt_a2.zero_grad()
                a2_loss.backward()
                p_opt_a2.step()

                curr_a1, log_pi1 = player_actor1.sample(s1)
                q1_pi_a, q1_pi_b = player_critic1(s1, curr_a1)
                a1_loss = (p_alpha * log_pi1 - torch.min(q1_pi_a, q1_pi_b)).mean()
                p_opt_a1.zero_grad()
                a1_loss.backward()
                p_opt_a1.step()

                # Soft Update
                for p, tp in zip(player_critic1.parameters(), player_critic1_target.parameters()):
                    tp.data.copy_(p_tau * p.data + (1 - p_tau) * tp.data)
                for p, tp in zip(player_critic2.parameters(), player_critic2_target.parameters()):
                    tp.data.copy_(p_tau * p.data + (1 - p_tau) * tp.data)

            # 2. 更新 Host RDQN
            if len(host_buffer) > batch_size:
                b_curr, b_hist, b_acts, b_rew, b_ncurr, b_nhist, b_done = host_buffer.sample(batch_size, device)
                
                b_act_pub = torch.tensor([a["public_signal"] for a in b_acts]).view(-1, 1).to(device)
                b_act_privs = [torch.tensor([a["private_signals"][i] for a in b_acts]).view(-1, 1).to(device) for i in range(config.num_players)]

                q_pub, q_privs = host_rdqn(b_curr, b_hist)
                q_pub_val = q_pub.gather(1, b_act_pub)
                q_priv_vals = sum([q_privs[i].gather(1, b_act_privs[i]) for i in range(config.num_players)])
                total_q = q_pub_val + q_priv_vals 

                with torch.no_grad():
                    next_q_pub, next_q_privs = host_target_rdqn(b_ncurr, b_nhist)
                    max_next_q = next_q_pub.max(1)[0].unsqueeze(1) + sum([q.max(1)[0].unsqueeze(1) for q in next_q_privs])
                    target_q = b_rew + h_gamma * max_next_q * (1 - b_done)

                h_loss = F.mse_loss(total_q, target_q)
                h_opt.zero_grad()
                h_loss.backward()
                h_opt.step()
                
                for p, tp in zip(host_rdqn.parameters(), host_target_rdqn.parameters()):
                    tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

            # ==========================================
            # Episode 結束處理 & 存檔
            # ==========================================
            if terminated or truncated:
                if prev_host_curr is not None:
                    curr_tensor, hist_tensor = process_host_obs(env, device=device)
                    host_buffer.push(
                        prev_host_curr, prev_host_hist, prev_host_action, 
                        last_host_reward, curr_tensor, hist_tensor, done=1
                    )
                prev_host_curr, prev_host_hist, prev_host_action = None, None, None
                
                h_epsilon = max(h_eps_min, h_epsilon * h_eps_decay)
                episodes += 1
                
                print(f"✅ Ep {episodes:4d} | Player Avg Reward: {episode_p_reward/step:.2f} | Host Total Profit: {episode_h_reward:.2f} | Host Eps: {h_epsilon:.3f}")
                
                # 每 100 個 Episode 存檔
                if episodes % 100 == 0:
                    save_path = os.path.join(checkpoint_dir, f"marl_checkpoint_ep_{episodes}.pth")
                    torch.save({
                        'episode': episodes,
                        'player_actor1': player_actor1.state_dict(),
                        'player_actor2': player_actor2.state_dict(),
                        'host_rdqn': host_rdqn.state_dict()
                    }, save_path)
                    print(f"💾 模型已儲存至: {save_path}")

                obs, info = env.reset()
                step = 0
                episode_p_reward = 0
                episode_h_reward = 0

if __name__ == "__main__":
    train_both()