import torch
import torch.nn as nn
import numpy as np

class HostAgent(nn.Module):
    """
    Host Agent Neural Network.
    Observes the true door and players' bets to generate signals for each player.
    """
    def __init__(self, num_players: int = 10, num_doors: int = 4):
        """
        Initialize the Host Agent network.
        
        Args:
            num_players (int): Number of players in the game.
            num_doors (int): Number of doors in the game.
        """
        super(HostAgent, self).__init__()
        self.num_players = num_players
        self.num_doors = num_doors
        
        # Input: true_door (one-hot, 4) + player_bets (10) + bankrupt_mask (10) = 24
        input_dim = num_doors + num_players * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_players * num_doors)
        )
        
    def forward(self, true_door: int, player_bets: np.ndarray, bankrupt_mask: np.ndarray) -> np.ndarray:
        """
        Forward pass to generate signals for all players.
        
        Args:
            true_door (int): The index of the correct door.
            player_bets (np.ndarray): 1D array of length num_players containing bet amounts.
            bankrupt_mask (np.ndarray): 1D boolean array indicating bankrupt players.
            
        Returns:
            np.ndarray: Array of shape (num_players,) with chosen signals in [0, num_doors-1].
        """
        door_one_hot = np.zeros(self.num_doors, dtype=np.float32)
        if 0 <= true_door < self.num_doors:
            door_one_hot[true_door] = 1.0
            
        bets = player_bets.astype(np.float32)
        # Normalize bets relative to typical initial balance
        bets = bets / 1000.0 
        
        mask = bankrupt_mask.astype(np.float32)
        
        x = np.concatenate([door_one_hot, bets, mask])
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        logits = self.net(x_tensor)
        logits = logits.view(-1, self.num_players, self.num_doors)
        
        # Greedily pick the argmax for deterministic output during inference
        signals = torch.argmax(logits, dim=-1).squeeze(0).numpy()
        return signals

class PlayerAgent(nn.Module):
    """
    Player Agent Neural Network.
    Has two heads: one for betting based on balance, one for choosing based on signal.
    """
    def __init__(self, num_doors: int = 4):
        """
        Initialize the Player Agent networks.
        
        Args:
            num_doors (int): Number of doors in the game.
        """
        super(PlayerAgent, self).__init__()
        self.num_doors = num_doors
        
        # Betting network (Input: normalized balance)
        self.bet_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs fraction of balance to bet
        )
        
        # Choosing network (Input: signal one-hot)
        self.choice_net = nn.Sequential(
            nn.Linear(num_doors, 32),
            nn.ReLU(),
            nn.Linear(32, num_doors)
        )
        
    def get_bet(self, balance: int) -> int:
        """
        Determine how much to bet based on current balance.
        
        Args:
            balance (int): Current balance of the player.
            
        Returns:
            int: Bet amount. -1 if bankrupt.
        """
        if balance <= 0:
            return -1
            
        x = torch.tensor([balance / 1000.0], dtype=torch.float32)
        fraction = self.bet_net(x).item()
        
        bet_amount = int(round(fraction * balance))
        return bet_amount
        
    def get_choice(self, signal: int) -> int:
        """
        Determine which door to choose based on the received signal.
        
        Args:
            signal (int): Signal index received from the Host.
            
        Returns:
            int: Door choice index. -1 if signal is invalid.
        """
        if signal < 0 or signal >= self.num_doors:
            return -1
            
        sig_one_hot = np.zeros(self.num_doors, dtype=np.float32)
        sig_one_hot[signal] = 1.0
        x = torch.tensor(sig_one_hot, dtype=torch.float32)
        
        logits = self.choice_net(x)
        choice = torch.argmax(logits).item()
        return int(choice)
