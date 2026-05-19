import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_obs(obs_dict, device="cpu"):
    curr = torch.as_tensor(obs_dict["current"], dtype=torch.float32, device=device)
    hist = torch.as_tensor(obs_dict["history"], dtype=torch.float32, device=device)
    hist_flat = hist.view(hist.shape[0], -1)
    return torch.cat([curr, hist_flat], dim=1)


def encode_features(state_flat, num_doors=4, seq_len=50):
    n = state_flat.shape[0]
    hist_feature_dim = 7 + num_doors

    curr = state_flat[:, :5]
    hist = state_flat[:, 5:].view(n, seq_len, hist_feature_dim)

    alive_bal_bribe = curr[:, 0:3]
    pub_sig = curr[:, 3].long()
    priv_sig = curr[:, 4].long()

    pub_sig = torch.where(pub_sig < 0, num_doors, pub_sig)
    priv_sig = torch.where(priv_sig < 0, num_doors, priv_sig)

    pub_onehot = F.one_hot(pub_sig, num_classes=num_doors + 1).float()
    priv_onehot = F.one_hot(priv_sig, num_classes=num_doors + 1).float()
    curr_encoded = torch.cat([alive_bal_bribe, pub_onehot, priv_onehot], dim=1)

    choice = hist[:, :, 0].long()
    h_pub = hist[:, :, 1].long()
    h_priv = hist[:, :, 2].long()
    continuous_hist = hist[:, :, 3:]

    choice = torch.where(choice < 0, num_doors, choice)
    h_pub = torch.where(h_pub < 0, num_doors, h_pub)
    h_priv = torch.where(h_priv < 0, num_doors, h_priv)

    choice_oh = F.one_hot(choice, num_classes=num_doors + 1).float()
    h_pub_oh = F.one_hot(h_pub, num_classes=num_doors + 1).float()
    h_priv_oh = F.one_hot(h_priv, num_classes=num_doors + 1).float()

    hist_encoded = torch.cat([choice_oh, h_pub_oh, h_priv_oh, continuous_hist], dim=2)
    return curr_encoded, hist_encoded


class TransformerExtractor(nn.Module):
    def __init__(self, curr_dim, hist_dim, d_model=128, nhead=4, num_layers=2, seq_len=50):
        super().__init__()
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=256,
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
    def __init__(self, num_doors=4):
        super().__init__()
        self.num_doors = num_doors
        curr_dim = 2 * num_doors + 5
        hist_dim = 4 * num_doors + 7
        self.extractor = TransformerExtractor(curr_dim=curr_dim, hist_dim=hist_dim)
        self.mu_layer = nn.Linear(256, 1)
        self.log_std_layer = nn.Linear(256, 1)  # kept for checkpoint compatibility

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat, num_doors=self.num_doors)
        feat = self.extractor(curr_enc, hist_enc)
        return self.mu_layer(feat)


class BetActor(nn.Module):
    def __init__(self, num_doors=4):
        super().__init__()
        self.num_doors = num_doors
        curr_dim = 2 * num_doors + 5
        hist_dim = 4 * num_doors + 7
        self.extractor = TransformerExtractor(curr_dim=curr_dim, hist_dim=hist_dim)
        self.door_logits_layer = nn.Linear(256, num_doors)
        self.bet_mu_layer = nn.Linear(256, 1)
        self.bet_log_std_layer = nn.Linear(256, 1)  # kept for checkpoint compatibility

    def forward(self, state_flat):
        curr_enc, hist_enc = encode_features(state_flat, num_doors=self.num_doors)
        feat = self.extractor(curr_enc, hist_enc)
        door_logits = self.door_logits_layer(feat)
        bet_mu = self.bet_mu_layer(feat)
        return door_logits, bet_mu


class TrainedPlayerAgent:
    def __init__(self, model_path: str, num_doors: int | None = None, device: str = "cpu"):
        self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device)
        ckpt_num_doors = checkpoint.get("num_doors")
        if num_doors is not None:
            resolved_num_doors = num_doors
        elif ckpt_num_doors is not None:
            resolved_num_doors = ckpt_num_doors
        else:
            resolved_num_doors = 4
        self.num_doors = int(resolved_num_doors)

        self.actor_bribe = BribeActor(num_doors=self.num_doors).to(self.device)
        self.actor_bet = BetActor(num_doors=self.num_doors).to(self.device)

        actor1_key = "player_actor1" if "player_actor1" in checkpoint else "actor1_state_dict"
        actor2_key = "player_actor2" if "player_actor2" in checkpoint else "actor2_state_dict"

        self.actor_bribe.load_state_dict(checkpoint[actor1_key])
        self.actor_bet.load_state_dict(checkpoint[actor2_key])

        self.actor_bribe.eval()
        self.actor_bet.eval()

        print(
            f"✅ PlayerAgent loaded (Episode {checkpoint.get('episode', 'Unknown')}, Doors={self.num_doors}, Device={self.device})"
        )

    def get_bribe_action(self, obs_dict: dict, deterministic: bool = True) -> np.ndarray:
        state_tensor = flatten_obs(obs_dict, device=self.device)
        with torch.no_grad():
            mu = self.actor_bribe(state_tensor)
            action = torch.sigmoid(mu)
        return action.cpu().numpy().flatten()

    def get_bet_action(self, obs_dict: dict, deterministic: bool = True) -> tuple[np.ndarray, np.ndarray]:
        state_tensor = flatten_obs(obs_dict, device=self.device)
        with torch.no_grad():
            door_logits, bet_mu = self.actor_bet(state_tensor)
            if deterministic:
                door_idx = torch.argmax(door_logits, dim=-1)
                bet_fraction = torch.sigmoid(bet_mu)
            else:
                door_probs = F.softmax(door_logits, dim=-1)
                door_idx = torch.multinomial(door_probs, num_samples=1).squeeze(-1)
                bet_fraction = torch.sigmoid(bet_mu)
        return door_idx.cpu().numpy(), bet_fraction.cpu().numpy().flatten()
