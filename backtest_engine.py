import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
data = pd.read_csv('features_full_daily.csv')

# --- AI-inspired signal logic ---
# Simple crossover logic: BUY when EMA21 > EMA50, SELL otherwise
data['signal'] = (data['ema21'] > data['ema50']).astype(int)

# Calculate daily returns and strategy returns
data['return'] = data['price'].pct_change()
data['strategy'] = data['signal'].shift(1) * data['return']

# --- Backtest Results ---
data['cumulative_market'] = (1 + data['return']).cumprod()
data['cumulative_strategy'] = (1 + data['strategy']).cumprod()

# --- Plot the performance comparison ---
plt.figure(figsize=(10, 6))
plt.plot(data['cumulative_market'], label='Market (XAUUSD)')
plt.plot(data['cumulative_strategy'], label='AI Strategy', linewidth=2)
plt.title('AI Strategy vs Market (Backtest Performance)')
plt.xlabel('Time')
plt.ylabel('Growth (x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('strategy_performance.png', dpi=300)

# --- Performance Summary ---
total_return = data['cumulative_strategy'].iloc[-1] - 1
market_return = data['cumulative_market'].iloc[-1] - 1

print(f"✅ AI Strategy Total Return: {total_return:.2%}")
print(f"📈 Market Total Return: {market_return:.2%}")
print("📊 Backtest completed and saved as 'strategy_performance.png' + 'backtest_results.csv'")

# --- Export results ---
data.to_csv('backtest_results.csv', index=False)

