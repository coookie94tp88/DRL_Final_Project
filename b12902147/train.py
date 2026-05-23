import os
import yaml
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import shutil
import argparse
import datetime

from env import DooRLEnv
from agents import HostAgent, PlayerAgent

def train(config_path="config/default_config.yaml", resume_ckpt=None, run_name=None):
    """
    Main training loop for DooRL.
    Supports Logging, Checkpointing, and Resuming from Checkpoint.
    
    Args:
        config_path (str): Path to the configuration YAML file.
        resume_ckpt (str, optional): Path to a checkpoint file to resume training.
        run_name (str, optional): Name for the training run. Defaults to timestamp.
    """
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpointing", {})
    
    num_players = env_cfg.get("num_players", 10)
    num_doors = env_cfg.get("num_doors", 4)
    total_timesteps = train_cfg.get("total_timesteps", 1000000)
    learning_rate = train_cfg.get("learning_rate", 3e-4)
    
    save_interval = ckpt_cfg.get("save_interval", 1000)
    
    # Create run-specific directories
    base_ckpt_dir = ckpt_cfg.get("checkpoint_dir", "./checkpoints/")
    base_log_dir = ckpt_cfg.get("log_dir", "./logs/")
    
    ckpt_dir = os.path.join(base_ckpt_dir, run_name)
    log_dir = os.path.join(base_log_dir, run_name)
    
    # Ensure directories exist
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup TensorBoard Logging
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[{run_name}] TensorBoard logs saved to {log_dir}")
    print(f"[{run_name}] Checkpoints saved to {ckpt_dir}")
    
    # Initialize Environment and Agents
    env = DooRLEnv(config_path)
    host = HostAgent(num_players, num_doors)
    players = [PlayerAgent(num_doors) for _ in range(num_players)]
    
    # Optimizers
    host_opt = torch.optim.Adam(host.parameters(), lr=learning_rate)
    player_opts = [torch.optim.Adam(p.parameters(), lr=learning_rate) for p in players]
    
    start_episode = 0
    
    # Resume Training Support
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming training from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt)
        
        host.load_state_dict(ckpt['host_state_dict'])
        host_opt.load_state_dict(ckpt['host_optimizer_state_dict'])
        
        for i, p in enumerate(players):
            p.load_state_dict(ckpt[f'player_{i}_state_dict'])
            player_opts[i].load_state_dict(ckpt[f'player_{i}_optimizer_state_dict'])
            
        start_episode = ckpt['episode']
        print(f"Successfully resumed at episode {start_episode}")
    else:
        # Save a backup of the configuration if starting fresh
        backup_path = os.path.join(ckpt_dir, "config_backup.yaml")
        shutil.copy(config_path, backup_path)
        print(f"Saved configuration backup to {backup_path}")
        
    # Convert timesteps to episodes based on rounds per episode
    episodes = total_timesteps // env.num_rounds
    
    print(f"Starting training loop for {episodes - start_episode} episodes...")
    for ep in range(start_episode, episodes):
        state = env.reset()
        done = False
        
        # Episode metrics
        ep_bets_placed = []
        ep_balances_when_betting = []
        ep_wins = 0
        ep_valid_choices = 0
        
        while not done:
            # --- Phase I: Betting ---
            env.phase_1_betting_start()
            
            bets = np.zeros(num_players, dtype=int)
            for i in range(num_players):
                # Bankrupt logic is handled inside get_bet and environment
                balance = state["player_balances"][i]
                bets[i] = players[i].get_bet(balance)
                if balance > 0:
                    ep_bets_placed.append(bets[i])
                    ep_balances_when_betting.append(balance)
                
            host_obs = env.phase_1_betting_step(bets)
            
            # --- Phase II: Signaling ---
            signals = host(host_obs["true_door"], host_obs["player_bets"], host_obs["bankrupt_mask"])
            player_obs = env.phase_2_signaling_step(signals)
            
            # --- Phase III: Choosing & Settlement ---
            choices = np.zeros(num_players, dtype=int)
            for i in range(num_players):
                if state["bankrupt_mask"][i]:
                    choices[i] = -1
                else:
                    choices[i] = players[i].get_choice(player_obs["signals"][i])
                    
            state, done, info = env.phase_3_choosing_step(choices)
            
            # Record metrics
            for i in range(num_players):
                if choices[i] != -1:
                    ep_valid_choices += 1
                    if choices[i] == info["true_door"]:
                        ep_wins += 1
            
            # Note: RL backward passes (loss computation, opt.step()) would go here.
            # They are omitted as we are focusing on providing the architectural pipeline.
            
        # Compute Episode Metrics
        active_players = int(np.sum(~state["bankrupt_mask"]))
        bankrupt_players = int(np.sum(state["bankrupt_mask"]))
        
        avg_bet = np.mean(ep_bets_placed) if ep_bets_placed else 0
        
        # Betting rate: bet / balance
        bet_rates = [b/bal if bal > 0 else 0 for b, bal in zip(ep_bets_placed, ep_balances_when_betting)]
        avg_bet_rate = np.mean(bet_rates) if bet_rates else 0
        
        win_rate = ep_wins / ep_valid_choices if ep_valid_choices > 0 else 0
        
        avg_final_balance = np.mean(state["player_balances"])
        
        # Logging Episode Statistics to TensorBoard
        writer.add_scalar("Metrics/Host_Total_Profit", state["host_balance"], ep)
        writer.add_scalar("Metrics/Avg_Final_Balance", avg_final_balance, ep)
        writer.add_scalar("Metrics/Active_Players", active_players, ep)
        writer.add_scalar("Metrics/Bankrupt_Players", bankrupt_players, ep)
        writer.add_scalar("Metrics/Avg_Bet_Amount", avg_bet, ep)
        writer.add_scalar("Metrics/Avg_Bet_Rate", avg_bet_rate, ep)
        writer.add_scalar("Metrics/Win_Rate", win_rate, ep)
        
        # Print info periodically to terminal (e.g. every 100 episodes)
        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1:05d} | Host Profit: {state['host_balance']:+6d} | Avg Bal: {avg_final_balance:6.1f} | "
                  f"Win Rate: {win_rate:.1%} | Bet Rate: {avg_bet_rate:.1%} | Actives: {active_players}/{num_players}")
        
        # Checkpointing
        if (ep + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep_{ep+1}.pt")
            save_dict = {
                'episode': ep + 1,
                'host_state_dict': host.state_dict(),
                'host_optimizer_state_dict': host_opt.state_dict(),
            }
            for i in range(num_players):
                save_dict[f'player_{i}_state_dict'] = players[i].state_dict()
                save_dict[f'player_{i}_optimizer_state_dict'] = player_opts[i].state_dict()
                
            torch.save(save_dict, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DooRL Training Pipeline")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--name", type=str, default=None, help="Name for the training run (used for logging and checkpoints)")
    
    args = parser.parse_args()
    train(
        config_path=args.config, 
        resume_ckpt=args.resume_ckpt, 
        run_name=args.name
    )
