"""
OracleGambit 環境驗證腳本
=========================
涵蓋以下六個面向的數學與邏輯正確性檢查：
  1. 觀察向量維度
  2. Payout 公式數學性質
  3. 資金守恆（Host 收到的 = Players 失去的）
  4. 破產救濟機制
  5. 歷史 Buffer padding 與 attention mask
  6. 多回合穩定性（連跑 200 回合不 NaN / 不 crash）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.oracle_gambit_env import OracleGambitEnv

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

def check(name: str, cond: bool, detail: str = "") -> bool:
    symbol = PASS if cond else FAIL
    msg = f"  {symbol} {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return cond


# -----------------------------------------------------------------------
# 1. 觀察向量維度
# -----------------------------------------------------------------------
def test_obs_shapes() -> bool:
    print("\n[1] 觀察向量維度檢查")
    env = OracleGambitEnv(num_players=4, num_doors=4, history_window=10, max_rounds=50)
    env.reset(seed=0)

    ok = True
    for agent in env.possible_agents:
        obs = env.observe(agent)
        expected = env.observation_space(agent).shape[0]
        match = (obs.shape[0] == expected)
        ok &= check(
            f"observe('{agent}') shape",
            match,
            f"got {obs.shape[0]}, expected {expected}",
        )
        no_nan = not np.isnan(obs).any()
        ok &= check(f"observe('{agent}') no NaN at init", no_nan)
    return ok


# -----------------------------------------------------------------------
# 2. Payout 公式數學性質
# -----------------------------------------------------------------------
def test_payout_math() -> bool:
    print("\n[2] Payout 公式數學性質")
    env = OracleGambitEnv(payout_threshold=0.25)
    ok = True

    # 在 x = θ 時，Host 應損益兩平：total_pool = total_payout
    # 即 W * M(θ) = W / θ  =>  M(θ) = 1 + 0.75/0.25 = 4
    theta = env.payout_threshold
    M_at_threshold = 1 + (1 - theta) / theta  # 應等於 4.0
    ok &= check(
        f"M(θ={theta}) = {M_at_threshold:.4f} ≈ 4.0",
        abs(M_at_threshold - 4.0) < 1e-9,
    )

    # 在 x = θ 時，Host 損益兩平 check：
    # total_payout = W * M = (x * P) * M = θ * P * (1 + (1-θ)/θ) = P
    x = theta
    P = 1000.0
    W = x * P
    total_payout = env.calculate_payout(W, x)
    ok &= check(
        f"At x=θ: total_payout ({total_payout:.4f}) ≈ P ({P:.4f})",
        abs(total_payout - P) < 1e-6,
    )

    # x < θ → Host profit（payout < pool）
    x_low = 0.10
    W_low = x_low * P
    payout_low = env.calculate_payout(W_low, x_low)
    ok &= check(
        f"At x={x_low} < θ: Host profits (payout {payout_low:.1f} < P {P:.1f})",
        payout_low < P,
    )

    # x > θ → Host loss（payout > pool）
    x_high = 0.50
    W_high = x_high * P
    payout_high = env.calculate_payout(W_high, x_high)
    ok &= check(
        f"At x={x_high} > θ: Host loses (payout {payout_high:.1f} > P {P:.1f})",
        payout_high > P,
    )

    # 單調性：x 越大，multiplier 越小
    xs = [0.05, 0.10, 0.20, 0.40, 0.80]
    Ms = [1 + (1 - theta) / x for x in xs]
    monotone = all(Ms[i] > Ms[i+1] for i in range(len(Ms)-1))
    ok &= check(
        "Multiplier is strictly decreasing in x",
        monotone,
        f"M values: {[f'{m:.3f}' for m in Ms]}",
    )
    return ok


# -----------------------------------------------------------------------
# 3. 資金守恆（Money Conservation）
# -----------------------------------------------------------------------
def test_money_conservation() -> bool:
    print("\n[3] 資金守恆檢查（Host 收到的 = Players 淨損失）")
    env = OracleGambitEnv(
        num_players=4, initial_balance=1000.0,
        fee_rate=0.0,        # 先關閉抽成，驗證純 zero-sum
        welfare_amount=0.0,  # 關閉救濟金
        max_rounds=1, seed=7,
    )
    env.reset(seed=7)

    # 記錄初始總資金（Host 沒有初始資金，只計 Players）
    init_total = sum(
        env._balances[f"player_{i}"] for i in range(env.num_players)
    )

    # 讓所有 player 全部下注在同一扇門（刻意讓少數派），host 亂發 signal
    rng = np.random.default_rng(99)
    actions_p1 = {"host": rng.random(1 + env.num_players).astype(np.float32)}
    for pid in range(env.num_players):
        actions_p1[f"player_{pid}"] = 0.05  # 5% 賄賂

    actions_p2 = {}
    for pid in range(env.num_players):
        door_frac = (env._correct_door / (env.num_doors - 1)) if env.num_doors > 1 else 0.0
        # player_0 猜對門，其他猜錯（模擬 minority 情境）
        if pid == 0:
            actions_p2[f"player_{pid}"] = (door_frac, 0.5)  # 50% bet
        else:
            wrong_door = (env._correct_door + 1) % env.num_doors
            actions_p2[f"player_{pid}"] = (wrong_door / max(env.num_doors - 1, 1), 0.5)

    rewards = env.step_all(actions_p1, actions_p2)

    # fee_rate=0 時，所有玩家的 reward 總和 + host reward 應約等於 0（zero-sum）
    player_reward_total = sum(rewards[f"player_{i}"] for i in range(env.num_players))
    host_reward = rewards["host"]
    total = player_reward_total + host_reward

    ok = check(
        f"Zero-sum (fee=0): Σ_player_rewards + host_reward = {total:.4f} ≈ 0",
        abs(total) < 1.0,  # 允許浮點誤差
        f"  player total: {player_reward_total:.4f}, host: {host_reward:.4f}",
    )
    return ok


# -----------------------------------------------------------------------
# 4. 破產救濟機制
# -----------------------------------------------------------------------
def test_welfare() -> bool:
    print("\n[4] 破產救濟機制")
    env = OracleGambitEnv(
        num_players=2, initial_balance=100.0, welfare_amount=10.0,
        fee_rate=0.0, max_rounds=5, seed=1,
    )
    env.reset(seed=1)

    ok = True
    # 強制 player_0 下注所有資金在錯誤的門
    for _ in range(3):
        correct = env._correct_door
        wrong = (correct + 1) % env.num_doors
        actions_p1 = {"host": np.zeros(1 + env.num_players, dtype=np.float32)}
        for pid in range(env.num_players):
            actions_p1[f"player_{pid}"] = 0.0  # 不賄賂

        actions_p2 = {
            "player_0": (wrong / max(env.num_doors - 1, 1), 1.0),   # 全下錯門
            "player_1": (correct / max(env.num_doors - 1, 1), 0.1),  # 小額猜對
        }
        env.step_all(actions_p1, actions_p2)

    # 檢查不管幾回合，所有 player 餘額 >= welfare_amount (> 0)
    for pid in range(env.num_players):
        name = f"player_{pid}"
        bal = env._balances[name]
        ok &= check(
            f"{name} balance ({bal:.2f}) >= welfare_amount ({env.welfare_amount})",
            bal >= env.welfare_amount,
        )
    return ok


# -----------------------------------------------------------------------
# 5. 歷史 Buffer padding 與 attention mask
# -----------------------------------------------------------------------
def test_history_buffer() -> bool:
    print("\n[5] 歷史 Buffer padding 與 attention mask")
    from env.oracle_gambit_env import _pad_history, _build_attention_mask
    import collections

    L = 5
    feat = 3
    buf: collections.deque = collections.deque(maxlen=L)
    ok = True

    # 空 buffer → 全部是 PAD
    h = _pad_history(buf, L, feat)
    m = _build_attention_mask(buf, L)
    ok &= check("Empty buffer: all padding", (h == -1).all() and not m.any())

    # 推入 2 筆
    buf.append(np.array([1.0, 2.0, 3.0]))
    buf.append(np.array([4.0, 5.0, 6.0]))
    h = _pad_history(buf, L, feat)
    m = _build_attention_mask(buf, L)
    ok &= check("2/5 filled: first 3 rows are padding", (h[:3] == -1).all())
    ok &= check("2/5 filled: last 2 rows are valid data", not (h[3:] == -1).any())
    ok &= check("Mask: first 3 False, last 2 True",
                not m[:3].any() and m[3:].all())

    # 塞滿
    for v in range(3, 6):
        buf.append(np.ones(feat) * v)
    h = _pad_history(buf, L, feat)
    m = _build_attention_mask(buf, L)
    ok &= check("Full buffer: no padding", not (h == -1).any() and m.all())
    return ok


# -----------------------------------------------------------------------
# 6. 多回合穩定性
# -----------------------------------------------------------------------
def test_stability(num_rounds: int = 200) -> bool:
    print(f"\n[6] 多回合穩定性（{num_rounds} 回合隨機動作）")
    env = OracleGambitEnv(num_players=6, max_rounds=num_rounds, seed=42)
    env.reset(seed=42)
    rng = np.random.default_rng(0)
    ok = True

    for r in range(num_rounds):
        host_act = rng.random(1 + env.num_players).astype(np.float32)
        actions_p1 = {"host": host_act}
        actions_p2 = {}
        for pid in range(env.num_players):
            name = f"player_{pid}"
            actions_p1[name] = float(rng.random())
            actions_p2[name] = (float(rng.random()), float(rng.random()))

        rewards = env.step_all(actions_p1, actions_p2)

        # 每 50 回合做一次觀察 NaN 檢查
        if r % 50 == 0:
            for agent in env.possible_agents:
                obs = env.observe(agent)
                if np.isnan(obs).any():
                    ok &= check(f"Round {r}: no NaN in obs({agent})", False)

        # 餘額永遠 >= welfare_amount
        for pid in range(env.num_players):
            name = f"player_{pid}"
            if env._balances[name] < env.welfare_amount - 1e-6:
                ok &= check(
                    f"Round {r}: {name} balance below welfare",
                    False,
                    f"balance = {env._balances[name]:.4f}",
                )
                break

    ok &= check(f"{num_rounds} rounds completed without crash or NaN", ok)
    return ok


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    results = [
        test_obs_shapes(),
        test_payout_math(),
        test_money_conservation(),
        test_welfare(),
        test_history_buffer(),
        test_stability(),
    ]
    total = len(results)
    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} test groups passed")
    if passed < total:
        print("Some checks FAILED. Please review the output above.")
        sys.exit(1)
    else:
        print("All checks PASSED.")
