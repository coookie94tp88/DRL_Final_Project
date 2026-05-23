import numpy as np
import random

class RandomPlayer:
    """
    A baseline player agent that bets a random fraction of its balance
    and chooses a random door regardless of the signal.
    """
    def __init__(self, num_doors: int = 4):
        self.num_doors = num_doors

    def get_bet(self, balance: int) -> int:
        """
        Returns a random integer bet between 0 and balance.
        """
        if balance <= 0:
            return -1
        return random.randint(0, balance)

    def get_choice(self, signal: int) -> int:
        """
        Ignores signal and picks a random door.
        """
        return random.randint(0, self.num_doors - 1)

class CautiousPlayer:
    """
    A baseline player that bets a small fixed fraction (e.g. 10%) of its balance
    and always trusts the signal provided by the host.
    """
    def __init__(self, num_doors: int = 4, bet_fraction: float = 0.1):
        self.num_doors = num_doors
        self.bet_fraction = bet_fraction

    def get_bet(self, balance: int) -> int:
        """
        Returns a small fraction of the balance as an integer.
        """
        if balance <= 0:
            return -1
        return int(round(balance * self.bet_fraction))

    def get_choice(self, signal: int) -> int:
        """
        Always follows the signal.
        """
        if signal < 0 or signal >= self.num_doors:
            return 0
        return signal

class HonestHost:
    """
    A baseline host that always tells the truth to every player.
    """
    def __init__(self, num_players: int = 10, num_doors: int = 4):
        self.num_players = num_players
        self.num_doors = num_doors

    def forward(self, true_door: int, player_bets: np.ndarray, bankrupt_mask: np.ndarray) -> np.ndarray:
        """
        Returns the true door for all players.
        """
        return np.full(self.num_players, true_door, dtype=int)

class DeceptiveHost:
    """
    A baseline host that never tells the truth. It picks a random incorrect door for each player.
    """
    def __init__(self, num_players: int = 10, num_doors: int = 4):
        self.num_players = num_players
        self.num_doors = num_doors

    def forward(self, true_door: int, player_bets: np.ndarray, bankrupt_mask: np.ndarray) -> np.ndarray:
        """
        Returns a random incorrect door for every player.
        """
        signals = np.zeros(self.num_players, dtype=int)
        for i in range(self.num_players):
            choices = [d for d in range(self.num_doors) if d != true_door]
            signals[i] = random.choice(choices)
        return signals
