import os
import time
import joblib
import pandas as pd
from loguru import logger
from drift_manager import detect_model_drift
from strategy_manager import generate_strategy_signal
from ensemble_train_retrain import retrain_if_drift_detected

# === CONFIG ===
MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/XAU_USD_Historical_Data_daily.csv"
REFRESH_INTERVAL = 300  # 5 minutes
DRIFT_CHECK_INTERVAL = 3600  # Every hour

# === INITIAL SETUP ===
os.makedirs("logs", exist_ok=True)
logger.add("logs/app_runtime.log", rotation="1 day", level="INFO")

def load_latest_model():
    """Load the latest available model."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠️ No trained model found, triggering training.")
        retrain_if_drift_detected(True)
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["features"]

def load_market_data():
    """Load and preprocess latest available data."""
    if not os.path.exists(DATA_PATH):
        logger.error("❌ No market data found for predictions.")
        return None
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        logger.warning("⚠️ Market data file is empty.")
        return None
    df.dropna(inplace=True)
    return df

def predict_and_signal(model, features, data):
    """Generate predictions and strategy signals."""
    try:
        if not all(f in data.columns for f in features):
            logger.warning("⚠️ Missing required features for prediction.")
            return None
        X = data[features].iloc[-1:].copy()
        pred = model.predict(X)[0]
        signal = "BUY" if pred == 1 else "SELL"
        logger.info(f"📊 Predicted Signal: {signal}")
        strategy_output = generate_strategy_signal(signal, data)
        return strategy_output
    except Exception as e:
        logger.error(f"⚠️ Prediction error: {e}")
        return None

def main():
    """Main runtime loop for the autonomous trading AI."""
    logger.info("🚀 Starting Autonomous AI Trading System ...")

    model, features = load_latest_model()
    drift_timer = 0

    while True:
        df = load_market_data()
        if df is not None:
            result = predict_and_signal(model, features, df)
            if result:
                logger.success(f"✅ Strategy Output: {result}")

        # Check for drift hourly
        drift_timer += REFRESH_INTERVAL
        if drift_timer >= DRIFT_CHECK_INTERVAL:
            drift_timer = 0
            logger.info("🔍 Running hourly drift detection...")
            accuracy = detect_model_drift(df)
            if accuracy and accuracy < 0.8:
                logger.warning("⚠️ Accuracy low — forcing retrain.")
                retrain_if_drift_detected(True)
                model, features = load_latest_model()
                logger.success("♻️ Reloaded updated ensemble model after retrain.")

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Graceful shutdown requested by user.")
    except Exception as e:
        logger.error(f"❌ Critical error in runtime: {e}")
