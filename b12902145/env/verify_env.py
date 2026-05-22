"""
OracleGambit (Simplified) — 環境驗證腳本
==========================================
涵蓋以下五個面向：
  1. 觀察向量維度
  2. Multiplier / Payout 公式數學性質
  3. Reward 守恆（zero-sum）
  4. 歷史 Buffer padding 與 attention mask
  5. 多回合穩定性（連跑 200 回合不 NaN / 不 crash）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.oracle_gambit_env import (
    OracleGambitEnv,
    _pad_history,
    _attention_mask,
)
import collections

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}" + (f"\n         {detail}" if detail else ""))
    return cond


# -----------------------------------------------------------------------
# 1. 觀察向量維度
# -----------------------------------------------------------------------
def test_obs_shapes() -> bool:
    print("\n[1] 觀察向量維度")
    env = OracleGambitEnv(num_players=4, num_doors=4, history_window=10, max_rounds=50)
    env.reset(seed=0)
    ok = True
    for agent in env.possible_agents:
        obs = env.observe(agent)
        expected = env.observation_space(agent).shape[0]
        ok &= check(
            f"observe('{agent}') shape",
            obs.shape[0] == expected,
            f"got {obs.shape[0]}, expected {expected}",
        )
        ok &= check(f"observe('{agent}') no NaN at init", not np.isnan(obs).any())
    return ok


# -----------------------------------------------------------------------
# 2. Multiplier 公式數學性質
# -----------------------------------------------------------------------
def test_payout_math() -> bool:
    print("\n[2] Multiplier 公式數學性質  (θ=0.20)")
    env = OracleGambitEnv(payout_threshold=0.20)
    θ = env.payout_threshold   # 0.20
    ok = True

    # M(θ) = 1 + (1-θ)/θ = 1 + 0.8/0.2 = 5.0
    M_at_θ = env.calculate_multiplier(θ)
    ok &= check(f"M(θ={θ}) = {M_at_θ:.4f} ≈ 5.0", abs(M_at_θ - 5.0) < 1e-9)

    # 在 x=θ 時, N*x*M(x) = N*θ*5 = N*(0.2*5) = N → break-even
    N = 10
    x = θ
    total_payout = N * x * env.calculate_multiplier(x)  # = N*θ*M(θ) = N
    ok &= check(
        f"At x=θ: total_payout({total_payout:.4f}) ≈ pool({float(N):.4f})",
        abs(total_payout - N) < 1e-9,
    )

    # x < θ → Host profits (payout < pool)
    x_low = 0.10
    payout_low = N * x_low * env.calculate_multiplier(x_low)
    ok &= check(
        f"At x={x_low} < θ: payout({payout_low:.2f}) < pool({N})",
        payout_low < N,
    )

    # x > θ → Host loses (payout > pool)
    x_high = 0.50
    payout_high = N * x_high * env.calculate_multiplier(x_high)
    ok &= check(
        f"At x={x_high} > θ: payout({payout_high:.2f}) > pool({N})",
        payout_high > N,
    )

    # 嚴格遞減
    xs = [0.05, 0.10, 0.20, 0.40, 0.80]
    Ms = [env.calculate_multiplier(x) for x in xs]
    ok &= check(
        "Multiplier strictly decreasing in x",
        all(Ms[i] > Ms[i+1] for i in range(len(Ms)-1)),
        f"M values: {[f'{m:.3f}' for m in Ms]}",
    )

    # M(0) → 0 (no winners → no payout)
    ok &= check("M(0) = 0.0", env.calculate_multiplier(0.0) == 0.0)

    return ok


# -----------------------------------------------------------------------
# 3. Reward 守恆 (zero-sum)
# -----------------------------------------------------------------------
def test_zero_sum() -> bool:
    print("\n[3] Reward 守恆 (zero-sum)")
    N = 6
    env = OracleGambitEnv(num_players=N, num_doors=4, payout_threshold=0.20, seed=7)
    env.reset(seed=7)
    rng = np.random.default_rng(999)
    ok = True

    for trial in range(50):
        host_act = rng.random()
        player_acts = {f"player_{i}": rng.random() for i in range(N)}
        rewards = env.step_all(host_act, player_acts)

        total = sum(rewards.values())
        # 由 reward 定義：Σ players + host = N*(M(x)-1)*x*N/N + N*(θ-x)*(-1)*(1-x)
        # 即 N*x*(M(x)-1) - N*(1-x) + N*(θ-x)
        # = N*(x*(1-θ)/x) - N*(1-x) + N*(θ-x)
        # = N*(1-θ) - N + N*x + N*θ - N*x
        # = N*(1-θ) - N + N*θ = N*(1 - θ + θ - 1) = 0 ✓
        ok &= check(
            f"trial {trial+1}: Σrewards = {total:.6f} ≈ 0",
            abs(total) < 1e-9,
        )
        if not ok:
            break

    return ok


# -----------------------------------------------------------------------
# 4. 歷史 Buffer padding 與 attention mask
# -----------------------------------------------------------------------
def test_history_buffer() -> bool:
    print("\n[4] 歷史 Buffer padding 與 attention mask")
    L, feat = 5, 3
    buf: collections.deque = collections.deque(maxlen=L)
    ok = True

    # 空 buffer → 全部 PAD
    h = _pad_history(buf, L, feat)
    m = _attention_mask(buf, L)
    ok &= check("Empty buf: all PAD", (h == -1).all() and not m.any())

    # 2 筆資料
    buf.append(np.array([1.0, 2.0, 3.0]))
    buf.append(np.array([4.0, 5.0, 6.0]))
    h = _pad_history(buf, L, feat)
    m = _attention_mask(buf, L)
    ok &= check("2/5: first 3 rows padding", (h[:3] == -1).all())
    ok &= check("2/5: last 2 rows valid",    not (h[3:] == -1).any())
    ok &= check("Mask: [F,F,F,T,T]",         not m[:3].any() and m[3:].all())

    # 塞滿
    for v in range(3, 6):
        buf.append(np.ones(feat) * v)
    h = _pad_history(buf, L, feat)
    m = _attention_mask(buf, L)
    ok &= check("Full: no padding, all True mask", not (h == -1).any() and m.all())

    # 透過 env 驗證實際觀測在多回合後 mask 正確增長
    env = OracleGambitEnv(num_players=2, history_window=10, max_rounds=20)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for step in range(5):
        env.step_all(rng.random(), {f"player_{i}": rng.random() for i in range(2)})
    # 5 rounds played → 5 valid slots at end of mask section
    obs_p = env.observe("player_0")
    L10 = 10
    feat_p = env._player_hist_feat
    hist_flat = obs_p[:L10 * feat_p]
    mask_flat = obs_p[L10 * feat_p: L10 * feat_p + L10].astype(bool)
    ok &= check("After 5 rounds: last 5 mask True",  mask_flat[-5:].all())
    ok &= check("After 5 rounds: first 5 mask False", not mask_flat[:5].any())

    return ok


# -----------------------------------------------------------------------
# 5. 多回合穩定性
# -----------------------------------------------------------------------
def test_stability(num_rounds: int = 200) -> bool:
    print(f"\n[5] 多回合穩定性 ({num_rounds} 回合)")
    N = 6
    env = OracleGambitEnv(num_players=N, max_rounds=num_rounds, seed=42)
    env.reset(seed=42)
    rng = np.random.default_rng(0)
    ok = True

    for r in range(num_rounds):
        host_act = rng.random()
        player_acts = {f"player_{i}": rng.random() for i in range(N)}
        try:
            rewards = env.step_all(host_act, player_acts)
        except Exception as e:
            return check(f"Round {r+1} crashed", False, str(e))

        no_nan = all(not np.isnan(v) for v in rewards.values())
        ok &= check(f"Round {r+1}: no NaN rewards", no_nan)
        if not no_nan:
            break

        for agent in env.possible_agents:
            obs = env.observe(agent)
            no_obs_nan = not np.isnan(obs).any()
            ok &= check(f"Round {r+1} {agent} obs no NaN", no_obs_nan)
            if not no_obs_nan:
                break
        if not ok:
            break

    return ok


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------
def main() -> None:
    results = [
        ("Observation Shapes",    test_obs_shapes()),
        ("Multiplier Math",       test_payout_math()),
        ("Zero-Sum Rewards",      test_zero_sum()),
        ("History Buffer",        test_history_buffer()),
        ("Stability (200 rounds)", test_stability()),
    ]

    print(f"\n{'═'*50}")
    print("  RESULTS")
    print(f"{'═'*50}")
    passed = 0
    for name, ok in results:
        tag = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
        print(f"  {tag}  {name}")
        passed += ok
    print(f"{'═'*50}")
    print(f"  {passed}/{len(results)} test groups passed")
    print(f"{'═'*50}\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
