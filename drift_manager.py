import os
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from datetime import datetime

# Paths
MODEL_PATH = "models/best_model.pkl"
BACKUP_PATH = "models/model_backup.pkl"
DATA_PATH = "data/XAU_USD_Historical_Data_daily.csv"
DRIFT_LOG = "logs/drift_log.csv"

# Threshold for accuracy drop to trigger rollback
DRIFT_THRESHOLD = 0.05  # 5% drop triggers backup restore

def detect_model_drift(new_data: pd.DataFrame):
    """Detects data or model drift, retrains or reverts if needed."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("❌ Model not found for drift check.")
        return None

    model = joblib.load(MODEL_PATH)

    # Create features similar to training
    df = new_data.copy()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr14"] = df["close"].pct_change().rolling(14).std()
    df.dropna(inplace=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    X = df[["ema21", "ema50", "atr14"]]
    y = df["target"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    logger.info(f"🧩 Drift check accuracy: {acc:.4f}")

    # Load historical accuracy benchmark
    prev_acc = None
    if os.path.exists(DRIFT_LOG):
        try:
            prev = pd.read_csv(DRIFT_LOG)
            prev_acc = float(prev.tail(1)["accuracy"].values[0])
        except Exception:
            prev_acc = None

    # Save latest accuracy to drift log
    new_entry = pd.DataFrame({
        "timestamp": [datetime.utcnow().isoformat()],
        "accuracy": [acc]
    })
    if os.path.exists(DRIFT_LOG):
        new_entry.to_csv(DRIFT_LOG, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(DRIFT_LOG, index=False)

    # If drop exceeds threshold, rollback
    if prev_acc and (prev_acc - acc) > DRIFT_THRESHOLD:
        logger.warning("⚠️ Model drift detected — rolling back to backup.")
        if os.path.exists(BACKUP_PATH):
            joblib.copy(BACKUP_PATH, MODEL_PATH)
            logger.success("✅ Model rollback successful.")
        else:
            logger.error("⚠️ No backup model found for rollback.")
    else:
        logger.info("✅ Model performing within normal range.")

    # Backup the stable model
    joblib.dump(model, BACKUP_PATH)
    logger.info("📦 Model backup updated.")
    return acc
