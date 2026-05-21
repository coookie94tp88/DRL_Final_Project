from env.oracle_gambit_env import OracleGambitEnv
import numpy as np

env = OracleGambitEnv(num_players=4, max_rounds=1000)
env.reset()
rng = np.random.default_rng(0)
total_host_profit = 0

for _ in range(1000):
    p1 = {"host": rng.random(1 + env.num_players)}
    p2 = {f"player_{i}": (rng.random(), rng.random()) for i in range(env.num_players)}
    rewards = env.step_all(p1, p2)
    total_host_profit += rewards["host"]

print(f"Host avg profit per round: {total_host_profit / 1000:.2f}")