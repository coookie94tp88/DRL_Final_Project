"""Load SB3 PPO player + PG host trained with train_sb3_proposal.py."""

from __future__ import annotations

import numpy as np
import torch

from proposal_vote_env import ProposalVoteConfig, ProposalVoteEnv
from train_sb3_proposal import PlayerExtractor, ProposalHostPolicy, host_obs_to_tensors


class SB3ProposalPlayerAgent:
    def __init__(self, model_path: str, device: str = "cpu"):
        from stable_baselines3 import PPO

        self.model = PPO.load(
            model_path,
            device=device,
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": PlayerExtractor,
                    "features_extractor_kwargs": {},
                }
            },
        )

    def get_vote_action(
        self, player_obs: dict, num_players: int, deterministic: bool = True
    ) -> np.ndarray:
        action, _ = self.model.predict(player_obs, deterministic=deterministic)
        return np.rint(np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)).astype(
            np.int32
        )[:num_players]


class SB3ProposalHostAgent:
    def __init__(
        self,
        checkpoint_path: str,
        config: ProposalVoteConfig | None = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        cfg = config or ProposalVoteConfig()
        tmp = ProposalVoteEnv(config=cfg)
        self.policy = ProposalHostPolicy(tmp.host_observation_space).to(self.device)
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.policy.eval()

    @torch.no_grad()
    def get_proposal(self, env: ProposalVoteEnv, deterministic: bool = True) -> int:
        host_obs = host_obs_to_tensors(env._get_observations()["host"])
        host_obs = {k: v.to(self.device) for k, v in host_obs.items()}
        logits = self.policy(host_obs)
        if deterministic:
            return int(logits.argmax().item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())
