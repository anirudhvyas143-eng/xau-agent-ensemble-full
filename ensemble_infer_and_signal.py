import os
import pandas as pd
import joblib
import datetime
import json
import numpy as np

# =========================================
# ⚡ XAUUSD AI Inference Engine
# =========================================
# Loads trained model + recent features
# Generates BUY/SELL signal with confidence
# For both hourly and daily inference use
# =========================================

# --- Configurable Paths ---
FEATURE_PATH = os.getenv("FEATURE_FILE", "features_full_daily.csv")
MODEL_PATH = os.getenv("MODEL_SAVE_PATH", "models/rf_model.joblib")

# --- Load Data ---
if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(f"❌ Feature file not found: {FEATURE_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

data = pd.read_csv(FEATURE_PATH)
model = joblib.load(MODEL_PATH)

# --- Extract features dynamically ---
feature_cols = [c for c in data.columns if c.startswith(("ema", "atr", "rsi", "volatility", "momentum"))]
if not feature_cols:
    feature_cols = ['ema21', 'ema50', 'atr14']

# --- Get the most recent valid row ---
last = data.dropna().iloc[-1]
X_latest = last[feature_cols].values.reshape(1, -1)

# --- Predict next move (1 = BUY, 0 = SELL) ---
prediction = int(model.predict(X_latest)[0])
probabilities = model.predict_proba(X_latest)[0]
confidence = float(np.max(probabilities))

# --- Current price ---
price = float(last['price'])

# --- Define entries & targets ---
if prediction == 1:
    signal = "BUY"
    conservative_entry = price * 0.998
    aggressive_entry = price * 1.002
    safer_entry = (conservative_entry + price) / 2
    take_profit = price * 1.008
    stop_loss = price * 0.994
else:
    signal = "SELL"
    conservative_entry = price * 1.002
    aggressive_entry = price * 0.998
    safer_entry = (conservative_entry + price) / 2
    take_profit = price * 0.992
    stop_loss = price * 1.006

# --- Prepare JSON Output ---
result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "timeframe": "daily",
    "signal": signal,
    "confidence": round(confidence, 4),
    "current_price": round(price, 2),
    "conservative_entry": round(conservative_entry, 2),
    "aggressive_entry": round(aggressive_entry, 2),
    "safer_entry": round(safer_entry, 2),
    "take_profit": round(take_profit, 2),
    "stop_loss": round(stop_loss, 2)
}

# --- Output JSON to console (Render logs / API bridge) ---
print(json.dumps(result, indent=2))

# --- Save last inference to history file ---
os.makedirs("history", exist_ok=True)
with open("history/inference_log.json", "a") as log:
    log.write(json.dumps(result) + "\n")

print(f"✅ Signal generated successfully at {result['timestamp']}")
