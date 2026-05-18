import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 載入你的環境設定
from env import OracleGambitConfig

class HostRDQN(nn.Module):
    """與 train_both.py 中完全一致的 RDQN 架構"""
    def __init__(self, num_players=10, num_doors=4, hist_dim=11, d_model=128):
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

class TrainedHostAgent:
    def __init__(self, checkpoint_path: str, config: OracleGambitConfig, hist_dim: int = 11, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.config = config
        
        # 初始化網路
        self.rdqn = HostRDQN(
            num_players=config.num_players, 
            num_doors=config.num_doors, 
            hist_dim=hist_dim
        ).to(self.device)
        
        # 載入權重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.rdqn.load_state_dict(checkpoint['host_rdqn'])
        self.rdqn.eval() # 切換到評估模式
        
        print(f"✅ HostAgent 載入完成: {checkpoint_path} (Device: {self.device})")

    def _process_obs(self, env):
        """將 env 當下的狀態轉為 Host 網路所需的 Tensor"""
        c = self.config
        winning_door = env.current_winning_door
        bribes = env.current_bribes
        
        win_door_oh = np.zeros(c.num_doors, dtype=np.float32)
        if winning_door >= 0:
            win_door_oh[winning_door] = 1.0
            
        curr_processed = np.concatenate([win_door_oh, bribes])
        curr_tensor = torch.FloatTensor(curr_processed).unsqueeze(0).to(self.device)
        
        # 提取 history
        hist = env._get_observations()["host"]["history"]
        hist_tensor = torch.FloatTensor(hist).unsqueeze(0).to(self.device)
        
        return curr_tensor, hist_tensor

    def get_action(self, env):
        """
        根據當前環境狀態推論 Host 動作。
        評估時我們採取貪婪策略 (Greedy)，直接選擇 Q 值最大的動作。
        """
        curr_tensor, hist_tensor = self._process_obs(env)
        
        with torch.no_grad():
            q_pub, q_privs = self.rdqn(curr_tensor, hist_tensor)
            
            pub_act = q_pub.argmax(dim=-1).item()
            priv_acts = [q.argmax(dim=-1).item() for q in q_privs]
            
        return pub_act, np.array(priv_acts, dtype=np.int32)