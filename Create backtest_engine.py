import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
data = pd.read_csv('features_full_daily.csv')

# Simple AI-based signal logic
data['signal'] = (data['ema21'] > data['ema50']).astype(int)
data['return'] = data['price'].pct_change()
data['strategy'] = data['signal'].shift(1) * data['return']

# Cumulative performance plot
(1 + data[['return', 'strategy']]).cumprod().plot(title="AI Strategy vs Market")

plt.xlabel('Time')
plt.ylabel('Growth (x)')
plt.tight_layout()
plt.savefig('strategy_performance.png')

# Summary
total_return = (data['strategy'] + 1).prod() - 1
print(f"✅ Strategy Total Return: {total_return:.2%}")

data.to_csv('backtest_results.csv', index=False)
