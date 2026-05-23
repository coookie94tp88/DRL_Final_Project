import os
import yaml
import numpy as np
from env import DooRLEnv
from baseline.agents import RandomPlayer, CautiousPlayer, HonestHost, DeceptiveHost

# ANSI Color Codes
class Colors:
    HEADER = '\033[0m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_test_env():
    """
    Initializes the DooRL environment with baseline agents to verify mechanics
    and prints formatted, colored logs for each phase without emojis.
    """
    print(f"{Colors.OKCYAN}Initializing Environment...{Colors.ENDC}")
    config_path = "config/default_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"{Colors.FAIL}Warning: {config_path} not found. Ensure you are running from the project root.{Colors.ENDC}")
        return
        
    env = DooRLEnv(config_path=config_path)
    
    # Initialize Baseline Agents
    host = HonestHost(num_players=env.num_players, num_doors=env.num_doors)
    
    # Players: Half random, half cautious
    players = []
    for i in range(env.num_players):
        if i % 2 == 0:
            players.append(RandomPlayer(num_doors=env.num_doors))
        else:
            players.append(CautiousPlayer(num_doors=env.num_doors, bet_fraction=0.15))
            
    print(f"{Colors.OKGREEN}Starting a new episode with Baseline Agents...{Colors.ENDC}")
    state = env.reset()
    done = False
    
    while not done:
        # Phase I: Betting Start
        env.phase_1_betting_start()
        
        # Collect bets
        bets = np.zeros(env.num_players, dtype=int)
        for i, p in enumerate(players):
            bets[i] = p.get_bet(state["player_balances"][i])
            
        host_obs = env.phase_1_betting_step(bets)
        
        # Phase II: Signaling
        signals = host.forward(
            host_obs["true_door"],
            host_obs["player_bets"],
            host_obs["bankrupt_mask"]
        )
        
        player_obs = env.phase_2_signaling_step(signals)
        
        # Phase III: Choosing
        choices = np.zeros(env.num_players, dtype=int)
        for i, p in enumerate(players):
            if state["player_balances"][i] <= 0:
                choices[i] = -1
            else:
                choices[i] = p.get_choice(player_obs["signals"][i])
                
        state, done, info = env.phase_3_choosing_step(choices)
        
        # Colorized Print Logic
        print(f"\n{Colors.HEADER}" + "="*80 + f"{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.OKCYAN}" + f"ROUND {state['current_round']:02d} / {env.num_rounds:02d}".center(80) + f"{Colors.ENDC}")
        print(f"{Colors.HEADER}" + "="*80 + f"{Colors.ENDC}")
        
        print(f"{Colors.OKCYAN}[Phase I] Betting{Colors.ENDC}")
        print(f"   Player Bets   : {host_obs['player_bets'].tolist()}")
        
        print(f"\n{Colors.OKCYAN}[Phase II] Signaling{Colors.ENDC}")
        print(f"   Host Signals  : {signals.tolist()}")
        
        print(f"\n{Colors.OKCYAN}[Phase III] Choosing & Results{Colors.ENDC}")
        print(f"   True Door     : {Colors.BOLD}{info['true_door']}{Colors.ENDC}")
        print(f"   Player Choices: {choices.tolist()}")
        print(f"   Player Rewards: {info['player_rewards'].tolist()}")
        
        print(f"\n{Colors.BOLD}[Settlement]{Colors.ENDC}")
        
        # Color Host Profit based on positive/negative
        host_profit_color = Colors.OKGREEN if info['host_reward'] >= 0 else Colors.FAIL
        print(f"   Host Profit   : {host_profit_color}{info['host_reward']:+d}{Colors.ENDC}")
        
        print(f"   Actives / Bankrupts: {info['active_players']} / {info['bankrupt_players']}")
        print(f"   Player Balances: {state['player_balances'].tolist()}")
        print(f"{Colors.HEADER}" + "="*80 + f"{Colors.ENDC}")

    print(f"\n{Colors.BOLD}{Colors.OKGREEN}Episode finished{Colors.ENDC}")
    final_host_color = Colors.OKGREEN if state['host_balance'] >= 0 else Colors.FAIL
    print(f"Final Host Balance: {final_host_color}{state['host_balance']}{Colors.ENDC}")

if __name__ == "__main__":
    run_test_env()
