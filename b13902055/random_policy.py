import numpy as np

# 假設你的環境檔案名稱為 env.py
from env import OracleGambitEnv, OracleGambitConfig, Phase

def run_random_agent():
    # 1. 初始化設定與環境 (為了測試快速，設定 max_rounds=10)
    config = OracleGambitConfig(
        num_players=10, 
        num_doors=4,
        max_rounds=100, 
        initial_balance=1000.0
    )
    env = OracleGambitEnv(config=config, seed=42)

    # 2. 重置環境
    obs, info = env.reset()
    terminated = False
    truncated = False

    print("=== OracleGambit 隨機測試開始 ===")
    
    # 3. 自定義的 Training Loop (狀態機驅動)
    while not terminated and not truncated:
        phase = env.phase # 也可以透過 info["phase"] 取得

        if phase == Phase.BRIBE:
            # 取得原生抽樣
            raw_bribes = env.player_bribe_action_space.sample()
            
            # 確保賄賂金額不會超過玩家目前餘額的 10% (這比較像正常的隨機玩家)
            max_affordable_bribes = env.balances * 0.1
            bribes = np.minimum(raw_bribes, max_affordable_bribes)
            
            action = {"player_bribes": bribes}
            obs, _, terminated, truncated, info = env.step(action)

        elif phase == Phase.SIGNAL:
            # [階段 2：主辦方給予訊號]
            host_action = env.host_action_space.sample()
            
            action = {
                "public_signal": host_action["public_signal"],
                "private_signals": host_action["private_signals"]
            }
            obs, _, terminated, truncated, info = env.step(action)
            # 註：這裡回傳的 reward 同樣是空字典 {}

        elif phase == Phase.BET:
            # [階段 3：玩家下注與結算]
            bet_action = env.player_bet_action_space.sample()
            
            action = {
                "player_doors": bet_action["doors"],
                "bet_fractions": bet_action["bet_fractions"]
            }
            # 只有這個階段會吐出真正的 rewards
            obs, rewards, terminated, truncated, info = env.step(action)

            # --- 印出這回合的結算狀況 ---
            print(f"Round {info['round']} 結算:")
            print(f"  👉 正確門牌: {info['winning_door']+1}")
            print(f"  👉 存活玩家數: {info['active_players']} / {config.num_players}")
            print(f"  👉 主辦方本局獲利: {rewards['host']:.2f}")
            print(f"  👉 玩家 0 的本局利潤: {rewards['players'][0]:.2f}")
            print(f"  📊 主辦方累計淨利: {info['host_cumulative_profit']:.2f}")
            
            # 印出玩家目前的餘額，觀察破產狀況
            print(f"  💰 玩家餘額: {np.round(env.balances, 1)}")
            print("-" * 40)

    print("=== 遊戲結束 ===")
    print(f"最終主辦方累計利潤: {info['host_cumulative_profit']:.2f}")
    if info['active_players'] == 0:
        print("所有玩家皆已破產！主辦方大獲全勝。")

if __name__ == "__main__":
    run_random_agent()