import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================
# ⚡ XAUUSD AI Strategy Backtest Engine
# ===============================================
# Evaluates EMA-based and AI-inspired crossover logic
# Generates backtest stats, performance plots, and CSV logs
# Works for daily / hourly datasets automatically
# ===============================================

# --- Configurable Data File ---
DATA_PATH = os.getenv("FEATURE_FILE", "features_full_daily.csv")
SAVE_DIR = "backtests"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Data ---
data = pd.read_csv(DATA_PATH)
if 'price' not in data.columns:
    raise ValueError("❌ Missing 'price' column in dataset!")

# --- AI-inspired Crossover Logic ---
data['signal'] = (data['ema21'] > data['ema50']).astype(int)
data['return'] = data['price'].pct_change()
data['strategy'] = data['signal'].shift(1) * data['return']

# --- Cumulative Performance ---
data['cumulative_market'] = (1 + data['return']).cumprod()
data['cumulative_strategy'] = (1 + data['strategy']).cumprod()

# --- Performance Metrics ---
def compute_metrics(df):
    returns = df['strategy'].dropna()
    market = df['return'].dropna()
    total_return = df['cumulative_strategy'].iloc[-1] - 1
    market_return = df['cumulative_market'].iloc[-1] - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
    max_drawdown = 1 - df['cumulative_strategy'] / df['cumulative_strategy'].cummax()
    mdd = max_drawdown.max()
    days = len(df)
    CAGR = ((1 + total_return) ** (252 / days)) - 1 if days > 0 else 0
    return {
        "total_return": total_return,
        "market_return": market_return,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "CAGR": CAGR
    }

metrics = compute_metrics(data)

# --- Save Plot ---
plt.figure(figsize=(12, 6))
plt.plot(data['cumulative_market'], label='Market (XAUUSD)', linestyle='--')
plt.plot(data['cumulative_strategy'], label='AI Strategy', linewidth=2)
plt.title('XAUUSD AI Strategy Backtest')
plt.xlabel('Time')
plt.ylabel('Growth (x)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, 'strategy_performance.png')
plt.savefig(plot_path, dpi=300)
plt.close()

# --- Export Data ---
csv_path = os.path.join(SAVE_DIR, 'backtest_results.csv')
data.to_csv(csv_path, index=False)

# --- Summary Report ---
print("✅ Backtest Completed")
print("📊 Summary Metrics:")
for k, v in metrics.items():
    print(f"   {k:15s}: {v:.2%}")

print(f"\n📈 Market Return: {metrics['market_return']:.2%}")
print(f"⚙️ Results saved to: {csv_path}")
print(f"🖼️ Chart saved to: {plot_path}")
