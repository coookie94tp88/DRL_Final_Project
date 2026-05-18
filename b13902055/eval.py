import numpy as np
import random
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

# 載入你的環境與自訂 Agent
from env import OracleGambitEnv, OracleGambitConfig, Phase
from player_agent import TrainedPlayerAgent  # 假設你將 agent 封裝存在 player_agent.py

def get_door_name(door_idx: int) -> str:
    """將數字 0, 1, 2, 3 轉換為 A, B, C, D"""
    if door_idx < 0: return "-"
    return chr(65 + int(door_idx))

def host_policy(bribes, winning_door, num_players, num_doors):
    """主辦方策略：賄賂前三名給真內線，其餘給隨機訊號"""
    public_signal = random.randint(0, num_doors - 1)
    private_signals = np.zeros(num_players, dtype=np.int32)
    
    # 找出賄賂金額前 3 名的 index
    top_3_indices = np.argsort(bribes)[-3:]
    
    for i in range(num_players):
        if i in top_3_indices and bribes[i] > 0:
            private_signals[i] = winning_door
        else:
            private_signals[i] = random.randint(0, num_doors - 1)
            
    return public_signal, private_signals

def render_round_log(console: Console, env: OracleGambitEnv, rewards: dict, info: dict, round_num: int):
    c = env.cfg
    
    correct_door = info["winning_door"]
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    choices = env.hist_choices[-1].astype(int)
    bribes = env.hist_bribes[-1]
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = rewards["host"]
    
    total_pool = np.sum(bets)
    door_totals = np.zeros(c.num_doors)
    for i in range(c.num_players):
        if bets[i] > 0:
            door_totals[choices[i]] += bets[i]
            
    # ── 1. 標頭 (Header) ──
    header = Text()
    header.append(f"Round   {round_num:2d}/{c.max_rounds}  ", style="cyan")
    header.append("Correct Door: ")
    header.append(f"🚪{get_door_name(correct_door)}  ", style="bold yellow")
    header.append("Public Signal: ")
    pub_style = "bold green" if pub_sig == correct_door else "bold red"
    pub_icon = "✔" if pub_sig == correct_door else "✖"
    header.append(f"🚪{get_door_name(pub_sig)}{pub_icon}  ", style=pub_style)
    host_color = "green" if host_profit >= 0 else "red"
    header.append(f"Host Reward: {host_profit:+.1f}", style=f"bold {host_color}")
    
    console.print(header)
    
    # ── 2. 下注分佈圖 (Betting Distribution) ──
    console.print(f"[grey74]Betting Distribution (total pool: {total_pool:.1f})[/grey74]")
    win_ratio = 0.0
    for d in range(c.num_doors):
        amt = door_totals[d]
        pct = (amt / total_pool * 100) if total_pool > 0 else 0
        
        bar_len = int((pct / 100) * 30)
        bar = "█" * bar_len
        bar_text = Text(f" 🚪{get_door_name(d)} │", style="grey74")
        
        bar_color = "green" if d == correct_door else "yellow"
        bar_text.append(f"{bar:<30}", style=bar_color)
        bar_text.append(f" {pct:5.1f}%  ({amt:6.1f})")
        
        if d == correct_door:
            bar_text.append(" ← ✔ CORRECT", style="bold yellow")
            win_ratio = pct / 100.0
            
        console.print(bar_text)
        
    threshold = c.payout_threshold
    profit_status = "Host profits" if win_ratio <= threshold else "Host loses"
    console.print(f"[grey74]Win-ratio x = {win_ratio:.3f}   threshold θ={threshold:.2f}  ({profit_status})[/grey74]")
    
    # ── 3. 玩家明細表 (Table) ──
    table = Table(show_header=True, header_style="grey74", box=None, padding=(0, 2))
    table.add_column("Player")
    table.add_column("PrivSig")
    table.add_column("Door")
    table.add_column("Bribe", justify="right")
    table.add_column("Bet", justify="right")
    table.add_column("Balance", justify="right")
    table.add_column("Reward", justify="right")
    
    for i in range(c.num_players):
        sig = priv_sigs[i]
        sig_str = f"🚪{get_door_name(sig)}"
        if sig == correct_door: sig_str += "✔"
        
        door = choices[i]
        door_str = f"🚪{get_door_name(door)}"
        if door == correct_door: door_str += "✔"
        
        rew = player_rewards[i]
        rew_str = f"[{'green' if rew > 0 else 'red'}]{rew:+.1f}[/]" if rew != 0 else "0.0"
        bal_str = f"[green]{env.balances[i]:.1f}[/green]"
        
        table.add_row(
            f"player_{i}", sig_str, door_str,
            f"{bribes[i]:.1f}", f"{bets[i]:.1f}", bal_str, rew_str
        )
        
    console.print(table)
    console.print(f"[grey74]Bribes total: {np.sum(bribes):.1f}[/grey74]")
    console.print("[cyan]" + "─"*70 + "[/cyan]\n")

def main():
    console = Console()
    
    # 初始化環境參數
    config = OracleGambitConfig(
        num_players=10, 
        num_doors=4, 
        max_rounds=20, 
        initial_balance=1000.0
    )
    env = OracleGambitEnv(config=config, seed=42)
    obs, info = env.reset()
    
    # ==========================================
    # 載入訓練好的 Agent
    # ==========================================
    model_path = "checkpoints/sac_checkpoint_ep_1600.pth"
    console.print(f"🔄 [cyan]正在載入訓練模型: {model_path}[/cyan]")
    try:
        agent = TrainedPlayerAgent(model_path=model_path, state_dim=555, num_doors=config.num_doors)
    except FileNotFoundError:
        console.print("[bold red]❌ 找不到模型檔案！請確認 checkpoints 資料夾中是否存在該檔案。[/bold red]")
        return

    console.print(Panel("OracleGambit - Trained Agent Evaluation\n"
                        f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}  Seed=42", 
                        style="bold yellow"))
    
    round_count = 1
    
    # ==========================================
    # 遊戲主迴圈
    # ==========================================
    while True:
        if env.phase == Phase.BRIBE:
            # 1. Agent 推論：決定賄賂比例 (使用 deterministic=True 確保穩定性)
            bribe_fractions = agent.get_bribe_action(obs["players"], deterministic=True)
            action = {"player_bribe_fractions": bribe_fractions}
            obs, _, _, _, info = env.step(action)
            
        elif env.phase == Phase.SIGNAL:
            # 2. 主辦方 (Host) 給出訊號：前三名給真內線
            winning_door = env.current_winning_door
            current_bribes = env.current_bribes
            pub_sig, priv_sigs = host_policy(current_bribes, winning_door, config.num_players, config.num_doors)
            
            action = {
                "public_signal": pub_sig,
                "private_signals": priv_sigs
            }
            obs, _, _, _, info = env.step(action)
            
        elif env.phase == Phase.BET:
            # 3. Agent 推論：決定選哪個門與下注比例
            doors, bet_fracs = agent.get_bet_action(obs["players"], deterministic=True)
            action = {
                "player_doors": doors,
                "bet_fractions": bet_fracs
            }
            obs, rewards, terminated, truncated, info = env.step(action)
            
            # 4. 渲染結果
            render_round_log(console, env, rewards, info, round_count)
            round_count += 1
            
            if terminated or truncated:
                console.print("[bold red]=== Game Over ===[/bold red]")
                break

if __name__ == "__main__":
    main()