import os
import shutil
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from datetime import datetime

# === CONFIG ===
MODEL_PATH = "models/best_model.pkl"
BACKUP_PATH = "models/model_backup.pkl"
DATA_PATH = "data/XAU_USD_Historical_Data_daily.csv"
DRIFT_LOG = "logs/drift_log.csv"
DRIFT_THRESHOLD = 0.05  # 5% accuracy drop triggers rollback

# === MAIN FUNCTION ===
def detect_model_drift(new_data: pd.DataFrame):
    """Detect and handle model drift — auto-rollback or backup if needed."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        logger.warning("❌ Model not found for drift check.")
        return None

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"⚠️ Error loading model: {e}")
        return None

    # === Build live features ===
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
    logger.info(f"🧩 Current model accuracy: {acc:.4f}")

    # === Load previous accuracy benchmark ===
    prev_acc = None
    if os.path.exists(DRIFT_LOG):
        try:
            prev = pd.read_csv(DRIFT_LOG)
            prev_acc = float(prev.tail(1)["accuracy"].values[0])
        except Exception:
            logger.warning("⚠️ Could not read previous drift log.")

    # === Save accuracy log ===
    new_entry = pd.DataFrame({
        "timestamp": [datetime.utcnow().isoformat()],
        "accuracy": [acc]
    })
    new_entry.to_csv(DRIFT_LOG, mode="a", header=not os.path.exists(DRIFT_LOG), index=False)

    # === Drift detection logic ===
    if prev_acc and (prev_acc - acc) > DRIFT_THRESHOLD:
        logger.warning(f"⚠️ Model drift detected! Drop: {(prev_acc - acc):.2%}")
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, MODEL_PATH)
            logger.success("✅ Model rollback completed using backup.")
        else:
            logger.error("❌ No backup found — rollback skipped.")
    else:
        logger.info("✅ Model accuracy within safe range.")

    # === Update stable model backup ===
    try:
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        logger.info("📦 Model backup refreshed successfully.")
    except Exception as e:
        logger.error(f"⚠️ Backup update failed: {e}")

    return acc

# === Test Run ===
if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        detect_model_drift(df)
    else:
        logger.warning("⚠️ No data available for drift test.")
