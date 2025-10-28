from flask import Flask, jsonify
import pandas as pd
import joblib
import json, threading, time
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

SIGNALS_FILE = Path("signals.json")
HISTORY_FILE = Path("signals_history.json")
MODEL_FILE = Path("model.pkl")
REFRESH_INTERVAL_SECS = 900  # 15 minutes


def train_default_model():
    """Train a simple fallback model if model.pkl is missing."""
    try:
        print("⚙️ Training default model (model.pkl not found)...")
        data = pd.read_csv('features_full_daily.csv')
        features = ['ema21', 'ema50', 'atr14']

        # Basic setup: target is whether ema21 > ema50
        data['target'] = (data['ema21'] > data['ema50']).astype(int)
        X = data[features].fillna(method='bfill')
        y = data['target']

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        print("✅ Default model trained and saved as model.pkl")
    except Exception as e:
        print(f"❌ Failed to train fallback model: {e}")


def generate_and_save_signal():
    """Core AI inference logic that generates, saves, and logs a signal."""
    try:
        # Ensure model exists
        if not MODEL_FILE.exists():
            train_default_model()

        model = joblib.load(MODEL_FILE)
        data = pd.read_csv('features_full_daily.csv')

        features = ['ema21', 'ema50', 'atr14']
        X = data[features].fillna(method='bfill')
        last_row = X.iloc[-1:]

        prediction = model.predict(last_row)[0]
        prob = model.predict_proba(last_row)[0][prediction]
        signal = "BUY" if prediction == 1 else "SELL"

        response = {
            "signal": signal,
            "confidence": round(prob * 100, 2),
            "timestamp": str(datetime.utcnow())
        }

        # Save latest signal
        with open(SIGNALS_FILE, "w") as f:
            json.dump(response, f, indent=2)

        # Append to history
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.load(open(HISTORY_FILE))
            except Exception:
                history = []
        history.append(response)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)

        print(f"[{response['timestamp']}] Signal: {signal} ({response['confidence']}%) — saved and logged")
        return response

    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return {"error": str(e)}


def background_scheduler():
    """Runs generate_and_save_signal every 15 min continuously."""
    while True:
        generate_and_save_signal()
        time.sleep(REFRESH_INTERVAL_SECS)


@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.utcnow())})


@app.route('/signal', methods=['GET'])
def signal_api():
    """Manually trigger signal generation (optional endpoint)."""
    result = generate_and_save_signal()
    return jsonify(result)


@app.route('/history', methods=['GET'])
def history_api():
    """View all historical signals."""
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
            return jsonify({"count": len(history), "history": history})
        except Exception as e:
            return jsonify({"error": f"Failed to read history: {e}"}), 500
    return jsonify({"message": "No signal history found"}), 404


if __name__ == '__main__':
    # Start background signal scheduler
    threading.Thread(target=background_scheduler, daemon=True).start()
    app.run(host='0.0.0.0', port=10000)
