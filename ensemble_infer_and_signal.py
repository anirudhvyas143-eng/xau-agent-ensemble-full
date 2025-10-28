import pandas as pd
import joblib
import datetime
import json

# Load historical feature dataset
data = pd.read_csv('features_full_daily.csv')

# Load trained Random Forest model
model = joblib.load('model.pkl')

# Get the latest row of data
last = data.iloc[-1]

# Extract feature columns
features = ['ema21', 'ema50', 'atr14']
X_latest = last[features].values.reshape(1, -1)

# Predict the next move (1 = BUY, 0 = SELL)
prediction = model.predict(X_latest)[0]
probability = model.predict_proba(X_latest)[0][prediction]

# Current price
current_price = float(last['price'])

# Define signal and entries
if prediction == 1:
    signal = "BUY"
    conservative_entry = current_price * 0.998
    aggressive_entry = current_price * 1.002
    safer_entry = (conservative_entry + current_price) / 2
    take_profit = current_price * 1.008
    stop_loss = current_price * 0.994
else:
    signal = "SELL"
    conservative_entry = current_price * 1.002
    aggressive_entry = current_price * 0.998
    safer_entry = (conservative_entry + current_price) / 2
    take_profit = current_price * 0.992
    stop_loss = current_price * 1.006

# Package result as JSON
result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "signal": signal,
    "confidence": round(float(probability), 4),
    "conservative_entry": round(conservative_entry, 2),
    "aggressive_entry": round(aggressive_entry, 2),
    "safer_entry": round(safer_entry, 2),
    "take_profit": round(take_profit, 2),
    "stop_loss": round(stop_loss, 2)
}

# Print formatted JSON output
print(json.dumps(result, indent=2))
