import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 網路架構 (必須與 train.py 中完全一致)
# ==========================================
class BribeActor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), 
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)

    def forward(self, state):
        feat = self.net(state)
        return self.mu_layer(feat) # 推論時只需要 mu

class BetActor(nn.Module):
    def __init__(self, state_dim, num_doors=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), 
            nn.Linear(256, 256), nn.ReLU()
        )
        self.door_logits_layer = nn.Linear(256, num_doors)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)

    def forward(self, state):
        feat = self.net(state)
        door_logits = self.door_logits_layer(feat)
        bet_mu = self.bet_mu_layer(feat)
        return door_logits, bet_mu

# ==========================================
# 2. 玩家 Agent 封裝類別
# ==========================================
class TrainedPlayerAgent:
    def __init__(self, model_path: str, state_dim: int = 555, num_doors: int = 4, device: str = "cpu"):
        """
        初始化並載入訓練好的模型權重
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        
        # 實例化網路
        self.actor_bribe = BribeActor(state_dim).to(self.device)
        self.actor_bet = BetActor(state_dim, num_doors).to(self.device)
        
        # 載入權重 (設定 map_location 確保在沒 GPU 的機器上也能跑)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor_bribe.load_state_dict(checkpoint['actor1_state_dict'])
        self.actor_bet.load_state_dict(checkpoint['actor2_state_dict'])
        
        # 設定為評估模式 (停用 Dropout/BatchNorm 行為)
        self.actor_bribe.eval()
        self.actor_bet.eval()
        
        print(f"✅ 成功載入模型 (Episode {checkpoint.get('episode', 'Unknown')})")

    def _flatten_obs(self, obs_dict: dict) -> torch.Tensor:
        """
        將環境傳來的玩家觀察字典攤平為 1D Tensor
        """
        curr = torch.FloatTensor(obs_dict["current"]).to(self.device)       # (N, 5)
        hist = torch.FloatTensor(obs_dict["history"]).to(self.device)       # (N, 50, 11)
        hist_flat = hist.view(hist.shape[0], -1)                            # (N, 550)
        return torch.cat([curr, hist_flat], dim=1)                          # (N, 555)

    def get_bribe_action(self, obs_dict: dict, deterministic: bool = True) -> np.ndarray:
        """
        取得賄賂階段的動作
        """
        state_tensor = self._flatten_obs(obs_dict)
        
        with torch.no_grad():
            mu = self.actor_bribe(state_tensor)
            
            if deterministic:
                # 測試時通常使用 deterministic (直接取平均值)，表現最穩定
                action = torch.sigmoid(mu)
            else:
                # 如果你想看它探索，可以加入常態分佈抽樣 (這裡簡化，直接回傳 mu 當參考)
                # 實務上測試時極少使用 stochastic 模式
                action = torch.sigmoid(mu) 
                
        return action.cpu().numpy().flatten()

    def get_bet_action(self, obs_dict: dict, deterministic: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        取得下注階段的動作 (回傳: 選擇的門, 下注比例)
        """
        state_tensor = self._flatten_obs(obs_dict)
        
        with torch.no_grad():
            door_logits, bet_mu = self.actor_bet(state_tensor)
            
            if deterministic:
                # 門：選擇機率最高的 (argmax)
                door_idx = torch.argmax(door_logits, dim=-1)
                # 下注比例：直接取 mu
                bet_fraction = torch.sigmoid(bet_mu)
            else:
                # 隨機抽樣 (通常測試時不需要)
                door_probs = F.softmax(door_logits, dim=-1)
                door_idx = torch.multinomial(door_probs, num_samples=1).squeeze(-1)
                bet_fraction = torch.sigmoid(bet_mu)
                
        return door_idx.cpu().numpy(), bet_fraction.cpu().numpy().flatten()