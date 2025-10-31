import os
import time
import threading
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger
from drift_manager import detect_model_drift
from strategy_manager import generate_strategy_signal
from ensemble_train_retrain import retrain_if_drift_detected

# ====================================================
# === CONFIGURATION ===
# ====================================================
MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/XAU_USD_Historical_Data_daily.csv"
REFRESH_INTERVAL = 300          # 5 minutes
DRIFT_CHECK_INTERVAL = 3600     # 1 hour

os.makedirs("logs", exist_ok=True)
logger.add("logs/app_runtime.log", rotation="1 day", level="INFO")

# ====================================================
# === FLASK APP SETUP ===
# ====================================================
app = Flask(__name__)
CORS(app)

latest_signal = {"signal": "N/A", "confidence": "N/A", "last_update": None}
model_accuracy = {"accuracy": "N/A"}
system_status = {"status": "Running", "last_retrain": None}

# ====================================================
# === CORE FUNCTIONS ===
# ====================================================
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

# ====================================================
# === BACKGROUND RUNTIME LOOP ===
# ====================================================
def background_trading_loop():
    """Continuously run AI-based predictions and drift detection."""
    global latest_signal, model_accuracy, system_status
    logger.info("🚀 Starting Autonomous AI Trading System ...")

    model, features = load_latest_model()
    drift_timer = 0

    while True:
        df = load_market_data()
        if df is not None:
            result = predict_and_signal(model, features, df)
            if result:
                latest_signal.update({
                    "signal": result.get("signal", "N/A"),
                    "confidence": result.get("confidence", "N/A"),
                    "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                logger.success(f"✅ Strategy Output: {result}")

        # Check for drift hourly
        drift_timer += REFRESH_INTERVAL
        if drift_timer >= DRIFT_CHECK_INTERVAL:
            drift_timer = 0
            logger.info("🔍 Running hourly drift detection...")
            accuracy = detect_model_drift(df)
            model_accuracy["accuracy"] = accuracy
            if accuracy and accuracy < 0.8:
                logger.warning("⚠️ Accuracy low — retraining triggered.")
                retrain_if_drift_detected(True)
                model, features = load_latest_model()
                system_status["last_retrain"] = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.success("♻️ Reloaded updated ensemble model after retrain.")

        time.sleep(REFRESH_INTERVAL)

# ====================================================
# === FLASK ROUTES ===
# ====================================================
@app.route("/")
def home():
    """Simple web dashboard."""
    html = f"""
    <html>
    <head><title>🟡 XAU/USD AI Agent Dashboard</title></head>
    <body style="font-family: Arial; background: #111; color: #ddd; text-align:center;">
        <h1>🤖 XAU/USD Autonomous AI Trading Agent</h1>
        <h3>Status: {system_status['status']}</h3>
        <p><b>Latest Signal:</b> {latest_signal['signal']}</p>
        <p><b>Confidence:</b> {latest_signal['confidence']}</p>
        <p><b>Last Update:</b> {latest_signal['last_update']}</p>
        <p><b>Model Accuracy:</b> {model_accuracy['accuracy']}</p>
        <p><b>Last Retrain:</b> {system_status['last_retrain']}</p>
        <hr>
        <p>🔄 Background engine running every {REFRESH_INTERVAL/60:.0f} minutes with hourly drift checks.</p>
        <p><a href='/retrain' style='color:yellow;'>Trigger Manual Retrain</a></p>
        <p><a href='/status' style='color:lightgreen;'>View JSON Status</a></p>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/status")
def status():
    """Return JSON status info."""
    return jsonify({
        "signal": latest_signal,
        "accuracy": model_accuracy,
        "system_status": system_status
    })

@app.route("/retrain")
def manual_retrain():
    """Trigger retraining manually."""
    global system_status
    retrain_if_drift_detected(True)
    system_status["last_retrain"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"message": "✅ Manual retraining complete", "time": system_status["last_retrain"]})

# ====================================================
# === MAIN ENTRYPOINT ===
# ====================================================
if __name__ == "__main__":
    # Start background loop in a separate thread
    t = threading.Thread(target=background_trading_loop, daemon=True)
    t.start()

    logger.info("🌐 Flask web server starting ...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
