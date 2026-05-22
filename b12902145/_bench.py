import time, sys
sys.path.insert(0, '.')
from env.oracle_gambit_env import OracleGambitEnv
from agents.transformer_agent import TransformerAgent
from training.ppo_runner import PPORunner

def bench(mb, epochs, label):
    env = OracleGambitEnv(num_players=6, num_doors=4)
    ha  = TransformerAgent(255, 4, hist_feat_size=4)
    pa  = TransformerAgent(455, 4, hist_feat_size=8)
    runner = PPORunner(env, ha, pa, ppo_epochs=epochs, minibatch_size=mb)
    env.reset()
    b = runner._collect_batch(32); runner._update(b)
    t0 = time.time()
    for _ in range(3):
        b = runner._collect_batch(128); runner._update(b)
    s = time.time() - t0
    rps = 128*3/s
    print(f"{label:36s}  {rps:6.1f} rds/s  (100k~{100000/rps/60:.0f}min)")

bench(mb=0,   epochs=4, label="full-batch 4ep  [before]")
bench(mb=256, epochs=4, label="mb=256     4ep")
bench(mb=256, epochs=2, label="mb=256     2ep  [recommended]")
bench(mb=128, epochs=2, label="mb=128     2ep")
