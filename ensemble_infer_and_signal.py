import os
import pandas as pd
import joblib
import datetime
import json
import numpy as np
from loguru import logger

# =========================================
# ⚡ XAUUSD AI ENSEMBLE INFERENCE ENGINE
# =========================================
# Multi-timeframe ensemble inference
# (daily + weekly + monthly)
# Outputs unified BUY/SELL + confidence
# =========================================

# --- Configurable Paths ---
DATA_DIR = "data"
MODEL_PATH = os.getenv("MODEL_SAVE_PATH", "models/ensemble_model.pkl")
OUTPUT_LOG = "history/inference_log.json"
os.makedirs("history", exist_ok=True)

# --- Logging Setup ---
logger.add("logs/inference.log", rotation="1 day", level="INFO")

# --- Load Ensemble Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

logger.info(f"📦 Loading ensemble model from {MODEL_PATH}")
loaded = joblib.load(MODEL_PATH)
model = loaded["model"]
features = loaded.get("features", [])

# --- Timeframe files ---
timeframes = {
    "daily": "features_full_daily.csv",
    "weekly": "features_full_weekly.csv",
    "monthly": "features_full_monthly.csv"
}

# --- Function to infer signal per timeframe ---
def infer_for_timeframe(tf_name, file_path):
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ Missing features for {tf_name} ({file_path})")
        return None
    
    df = pd.read_csv(file_path).dropna()
    if df.empty:
        logger.warning(f"⚠️ Empty data for {tf_name}")
        return None

    last = df.iloc[-1]
    X = last[features].values.reshape(1, -1)
    probs = model.predict_proba(X)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    price = float(last["Close"])
    
    if pred == 1:
        signal = "BUY"
        tp = price * 1.008
        sl = price * 0.994
    else:
        signal = "SELL"
        tp = price * 0.992
        sl = price * 1.006
    
    result = {
        "timeframe": tf_name,
        "signal": signal,
        "confidence": round(conf, 4),
        "price": round(price, 2),
        "take_profit": round(tp, 2),
        "stop_loss": round(sl, 2)
    }
    return result


# --- Run inference across all timeframes ---
results = []
for tf, fname in timeframes.items():
    path = os.path.join(DATA_DIR, fname)
    res = infer_for_timeframe(tf, path)
    if res:
        results.append(res)

if not results:
    raise RuntimeError("❌ No valid inference results. Check data files.")

# --- Weighted signal aggregation ---
signal_map = {"BUY": 1, "SELL": 0}
weights = {"daily": 0.5, "weekly": 0.3, "monthly": 0.2}

weighted_conf = 0
weighted_signal = 0
for r in results:
    s_val = signal_map[r["signal"]]
    weighted_signal += s_val * weights.get(r["timeframe"], 0)
    weighted_conf += r["confidence"] * weights.get(r["timeframe"], 0)

final_signal = "BUY" if weighted_signal >= 0.5 else "SELL"
final_confidence = round(weighted_conf, 4)
current_price = results[0]["price"]

# --- Entry & Targets ---
if final_signal == "BUY":
    conservative_entry = current_price * 0.998
    aggressive_entry = current_price * 1.002
    safer_entry = (conservative_entry + current_price) / 2
    take_profit = current_price * 1.008
    stop_loss = current_price * 0.994
else:
    conservative_entry = current_price * 1.002
    aggressive_entry = current_price * 0.998
    safer_entry = (conservative_entry + current_price) / 2
    take_profit = current_price * 0.992
    stop_loss = current_price * 1.006

# --- Final JSON Output ---
final_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "final_signal": final_signal,
    "final_confidence": final_confidence,
    "current_price": round(current_price, 2),
    "conservative_entry": round(conservative_entry, 2),
    "aggressive_entry": round(aggressive_entry, 2),
    "safer_entry": round(safer_entry, 2),
    "take_profit": round(take_profit, 2),
    "stop_loss": round(stop_loss, 2),
    "timeframe_breakdown": results
}

# --- Print JSON (Render log bridge) ---
print(json.dumps(final_result, indent=2))

# --- Append to log file ---
with open(OUTPUT_LOG, "a") as f:
    f.write(json.dumps(final_result) + "\n")

logger.success(f"✅ Inference complete — {final_signal} @ {current_price} | Confidence {final_confidence}")
