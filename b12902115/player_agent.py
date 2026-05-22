import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_encoding import flatten_obs, encode_features, STATE_DIM, CURR_DIM, HIST_DIM, SEQ_LEN


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
        self.extractor = TransformerExtractor()
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        mu = self.mu_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), self.log_std_min, self.log_std_max)
        return mu


class BetActor(nn.Module):
    NUM_BELIEFS = 3
    BELIEF_NAMES = ("pub", "priv", "rnd")

    def __init__(self, gumbel_tau=0.8, log_std_min=-5.0, log_std_max=1.0):
        super().__init__()
        self.gumbel_tau = gumbel_tau
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.extractor = TransformerExtractor()
        self.belief_logits_layer = nn.Linear(256, self.NUM_BELIEFS)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat)
        feat = self.extractor(curr_enc, hist_enc)
        return self.belief_logits_layer(feat), self.bet_mu_layer(feat)


class TrainedPlayerAgent:
    BELIEF_NAMES = BetActor.BELIEF_NAMES

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.actor_bribe = BribeActor().to(self.device)
        self.actor_bet = BetActor().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        actor1_key = "player_actor1" if "player_actor1" in checkpoint else "actor1_state_dict"
        actor2_key = "player_actor2" if "player_actor2" in checkpoint else "actor2_state_dict"
        self.actor_bribe.load_state_dict(checkpoint[actor1_key])
        self.actor_bet.load_state_dict(checkpoint[actor2_key])
        self.actor_bribe.eval()
        self.actor_bet.eval()
        print(f"✅ Player model loaded (Episode {checkpoint.get('episode', 'Unknown')})")

    def get_bribe_action(self, obs_dict: dict, deterministic: bool = True) -> np.ndarray:
        state_tensor = flatten_obs(obs_dict, device=self.device)
        with torch.no_grad():
            mu = self.actor_bribe(state_tensor)
            action = torch.sigmoid(mu)
        return action.cpu().numpy().flatten()

    def get_bet_action(self, obs_dict: dict, deterministic: bool = True) -> tuple[np.ndarray, np.ndarray]:
        state_tensor = flatten_obs(obs_dict, device=self.device)
        with torch.no_grad():
            belief_logits, bet_mu = self.actor_bet(state_tensor)
            if deterministic:
                belief_idx = torch.argmax(belief_logits, dim=-1)
                bet_fraction = torch.sigmoid(bet_mu)
            else:
                belief_probs = F.softmax(belief_logits, dim=-1)
                belief_idx = torch.multinomial(belief_probs, num_samples=1).squeeze(-1)
                bet_fraction = torch.sigmoid(bet_mu)
        return belief_idx.cpu().numpy(), bet_fraction.cpu().numpy().flatten()
