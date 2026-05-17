import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

# 載入你的環境
from env import OracleGambitEnv, OracleGambitConfig, Phase

def get_door_name(door_idx: int) -> str:
    """將數字 0, 1, 2, 3 轉換為 A, B, C, D"""
    if door_idx < 0: return "-"
    return chr(65 + int(door_idx))

def render_round_log(console: Console, env: OracleGambitEnv, rewards: dict, info: dict, round_num: int):
    c = env.cfg
    
    # 透過剛剛寫入的 history 抓取這一回合的「確實」數據
    correct_door = info["winning_door"]
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    choices = env.hist_choices[-1].astype(int)
    bribes = env.hist_bribes[-1]
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = rewards["host"]
    
    # 計算 Pool 與各門比例
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
    for d in range(c.num_doors):
        amt = door_totals[d]
        pct = (amt / total_pool * 100) if total_pool > 0 else 0
        
        # 畫長條圖 (最多 30 格)
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
        
    # 主辦方獲利判定
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
        # 標記 Private Signal 是否正確
        sig = priv_sigs[i]
        sig_str = f"🚪{get_door_name(sig)}"
        if sig == correct_door: sig_str += "✔"
        
        # 標記選擇的門是否正確
        door = choices[i]
        door_str = f"🚪{get_door_name(door)}"
        if door == correct_door: door_str += "✔"
        
        # 獎金顏色
        rew = player_rewards[i]
        rew_str = f"[{'green' if rew > 0 else 'red'}]{rew:+.1f}[/]" if rew != 0 else "0.0"
        
        # 餘額
        bal_str = f"[green]{env.balances[i]:.1f}[/green]"
        
        table.add_row(
            f"player_{i}",
            sig_str,
            door_str,
            f"{bribes[i]:.1f}",
            f"{bets[i]:.1f}",
            bal_str,
            rew_str
        )
        
    console.print(table)
    console.print(f"[grey74]Bribes total: {np.sum(bribes):.1f}[/grey74]")
    console.print("[cyan]" + "─"*70 + "[/cyan]\n")

def main():
    console = Console()
    
    # 設定參數 (縮小 max_bribe_fraction 防止第一回合就破產)
    config = OracleGambitConfig(
        num_players=5, 
        num_doors=4, 
        max_rounds=10, 
        initial_balance=1000.0,
        max_bribe_fraction=1 # 限制隨機策略最多拿 10% 出來賄賂
    )
    env = OracleGambitEnv(config=config, seed=42)
    obs, info = env.reset()
    
    console.print(Panel("OracleGambit - Random Agent Simulation\n"
                        f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}  Seed=42", 
                        style="bold yellow"))
    
    round_count = 1
    
    # 進行遊戲迴圈
    while True:
        if env.phase == Phase.BRIBE:
            action = {"player_bribe_fractions": env.player_bribe_action_space.sample()}
            env.step(action)
            
        elif env.phase == Phase.SIGNAL:
            action = env.host_action_space.sample()
            env.step(action)
            
        elif env.phase == Phase.BET:
            
            action = env.player_bet_action_space.sample()
            action["player_doors"] = action["doors"]
            obs, rewards, terminated, truncated, info = env.step(action)

            
            
            # 在這一局 (Round) 結束時，呼叫排版函式
            render_round_log(console, env, rewards, info, round_count)
            round_count += 1
            
            if terminated or truncated:
                console.print("[bold red]=== Game Over ===[/bold red]")
                break

if __name__ == "__main__":
    main()