import numpy as np
from typing import Tuple, Dict, Any
import yaml

def calculate_payouts(
    bets: np.ndarray, 
    choices: np.ndarray, 
    true_door: int, 
    theta: float = 0.25
) -> Tuple[np.ndarray, int]:
    """
    Calculate the financial settlement for all players and the host in a single round.
    Bankrupt players (indicated by choices == -1 or bets == -1) are ignored in the calculation.
    All financial metrics are calculated as integers.

    Args:
        bets (np.ndarray): A 1D array of length 10 containing the int bet amounts of each player.
                           Value is -1 for bankrupt players.
        choices (np.ndarray): A 1D array of length 10 containing the door index chosen by each player.
                              Value is -1 for bankrupt players.
        true_door (int): The index of the correct door (0, 1, 2, or 3).
        theta (float, optional): The break-even threshold for the host. Defaults to 0.25.

    Returns:
        Tuple[np.ndarray, int]: 
            - np.ndarray: A 1D int array of length 10 containing the net profit/loss for each player.
            - int: The net profit/loss for the host in this round.
    """
    
    # Filter out bankrupt players (active players have choice != -1)
    active_mask = (choices != -1)
    
    # Calculate total pool only from active players
    active_bets = bets * active_mask
    total_pool = int(np.sum(active_bets))
    
    if total_pool <= 0:
        # No valid bets placed this round
        return np.zeros_like(bets, dtype=int), 0
        
    # Identify winners
    winners_mask = (choices == true_door) & active_mask
    winning_volume = int(np.sum(bets[winners_mask]))
    
    # Calculate dynamic multiplier M(x)
    if winning_volume == 0:
        payout_multiplier = 0.0
    else:
        x_ratio = winning_volume / total_pool
        payout_multiplier = 1.0 + (1.0 - theta) / x_ratio
        
    # Calculate rewards arrays
    player_rewards = np.zeros_like(bets, dtype=int)
    total_payout = 0
    
    for i in range(len(bets)):
        if not active_mask[i]:
            continue # Bankrupt players get 0 reward update
            
        if winners_mask[i]:
            # Net profit = (Bet * Multiplier) - Original Bet
            gross_win = int(round(bets[i] * payout_multiplier))
            player_rewards[i] = gross_win - bets[i]
            total_payout += gross_win
        else:
            # Loss = Negative Original Bet
            player_rewards[i] = -bets[i]
            
    # Host gets total pool minus total payout given to winners
    host_reward = total_pool - total_payout
    
    return player_rewards, host_reward

class DooRLEnv:
    """
    DooRL Environment v2.0
    A pure Python RL environment without external dependencies (e.g., gym/gymnasium).
    Manages 1 Host and 10 Players over a specified number of rounds per episode.
    """
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        Initialize the DooRL environment with parameters from the config file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        env_cfg = config.get("environment", {})
        self.num_players = env_cfg.get("num_players", 10)
        self.num_doors = env_cfg.get("num_doors", 4)
        self.num_rounds = env_cfg.get("num_rounds_per_episode", 10)
        self.initial_balance = int(env_cfg.get("initial_balance", 1000))
        self.theta = env_cfg.get("theta", 0.25)
        
        # State variables
        self.player_balances = np.zeros(self.num_players, dtype=int)
        self.host_balance = 0
        self.current_round = 0
        
        # In-round state
        self.true_door = -1
        self.current_bets = np.zeros(self.num_players, dtype=int)
        self.current_signals = np.zeros(self.num_players, dtype=int)
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Returns:
            Dict[str, Any]: The initial state of the environment, including balances.
        """
        self.player_balances = np.full(self.num_players, self.initial_balance, dtype=int)
        self.host_balance = 0
        self.current_round = 0
        return self._get_global_state()

    def _get_global_state(self) -> Dict[str, Any]:
        """
        Get the current global state, including who is bankrupt.
        
        Returns:
            Dict[str, Any]: Global state dictionary.
        """
        return {
            "player_balances": self.player_balances.copy(),
            "host_balance": self.host_balance,
            "current_round": self.current_round,
            "bankrupt_mask": self.player_balances <= 0
        }

    def phase_1_betting_start(self) -> None:
        """
        Start Phase I: The environment selects the true door.
        Players should be queried for their bets after this.
        """
        self.true_door = np.random.randint(0, self.num_doors)
        self.current_bets = np.zeros(self.num_players, dtype=int)

    def phase_1_betting_step(self, bets: np.ndarray) -> Dict[str, Any]:
        """
        Process Phase I: Players place bets.
        
        Args:
            bets (np.ndarray): 1D array of length num_players. Bankrupt players must have -1.
        
        Returns:
            Dict[str, Any]: Observations for the Host (true door and all bets).
        """
        processed_bets = np.zeros(self.num_players, dtype=int)
        bankrupt_mask = self.player_balances <= 0
        
        for i in range(self.num_players):
            if bankrupt_mask[i]:
                # Force -1 for bankrupt players
                processed_bets[i] = -1
            else:
                # Clip bet to [0, current_balance]
                bet = max(0, min(int(bets[i]), self.player_balances[i]))
                processed_bets[i] = bet
                
        self.current_bets = processed_bets
        
        # Prepare observation for the Host
        host_obs = {
            "true_door": self.true_door,
            "player_bets": self.current_bets.copy(),
            "bankrupt_mask": bankrupt_mask.copy()
        }
        return host_obs

    def phase_2_signaling_step(self, signals: np.ndarray) -> Dict[str, Any]:
        """
        Process Phase II: Host provides signals to players.
        
        Args:
            signals (np.ndarray): 1D array of length num_players containing signals in {0,1,2,3}.
        
        Returns:
            Dict[str, Any]: Observations for each Player (their specific signal).
        """
        # Ensure signals are within valid bounds (0 to num_doors - 1)
        self.current_signals = np.clip(signals, 0, self.num_doors - 1).astype(int)
        
        player_obs = {
            "signals": self.current_signals.copy()
        }
        return player_obs

    def phase_3_choosing_step(self, choices: np.ndarray) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Process Phase III: Players choose a door based on signals. Settle rewards.
        
        Args:
            choices (np.ndarray): 1D array of length num_players. Bankrupt players must have -1.
        
        Returns:
            Tuple[Dict[str, Any], bool, Dict[str, Any]]:
                - State (balances, etc.)
                - Done (bool)
                - Info dictionary containing round statistics
        """
        bankrupt_mask = self.player_balances <= 0
        processed_choices = np.zeros(self.num_players, dtype=int)
        
        for i in range(self.num_players):
            if bankrupt_mask[i]:
                processed_choices[i] = -1
            else:
                choice = int(choices[i])
                if choice < 0 or choice >= self.num_doors:
                    # Invalid choices default to 0 to prevent index errors
                    choice = 0
                processed_choices[i] = choice

        # Settlement using the core engine
        player_rewards, host_reward = calculate_payouts(
            self.current_bets, 
            processed_choices, 
            self.true_door, 
            self.theta
        )
        
        # Update balances
        self.player_balances += player_rewards
        self.host_balance += host_reward
        
        # Advance round
        self.current_round += 1
        done = (self.current_round >= self.num_rounds)
        
        info = {
            "true_door": self.true_door,
            "player_rewards": player_rewards.copy(),
            "host_reward": host_reward,
            "active_players": int(np.sum(self.player_balances > 0)),
            "bankrupt_players": int(np.sum(self.player_balances <= 0))
        }
        
        return self._get_global_state(), done, info
