from flask import Flask, jsonify
import pandas as pd
import joblib
import json, threading, time, os
from datetime import datetime
from pathlib import Path
import numpy as np

app = Flask(__name__)

# === File paths ===
DAILY_FILE = Path("features_full_daily.csv")
WEEKLY_FILE = Path("features_full_weekly.csv")
MONTHLY_FILE = Path("features_full_monthly.csv")
MODEL_FILE = Path("model.pkl")
SIGNALS_FILE = Path("signals.json")
HISTORY_FILE = Path("signals_history.json")

REFRESH_INTERVAL_SECS = 900  # 15 minutes


# === Utility: compute EMA + ATR if missing ===
def ensure_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.sort_values('Date', inplace=True)
    df.dropna(subset=['Close'], inplace=True)

    if 'ema21' not in df.columns:
        df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    if 'ema50' not in df.columns:
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    if 'atr14' not in df.columns:
        df['H-L'] = df['High'] - df['Low']
        df['H-C'] = (df['High'] - df['Close'].shift()).abs()
        df['L-C'] = (df['Low'] - df['Close'].shift()).abs()
        tr = df[['H-L', 'H-C', 'L-C']].max(axis=1)
        df['atr14'] = tr.rolling(window=14).mean()

    df.dropna(inplace=True)
    return df


def get_signal_from_trend(df):
    """Return simple BUY/SELL based on EMA cross."""
    if df["ema21"].iloc[-1] > df["ema50"].iloc[-1]:
        return "BUY"
    else:
        return "SELL"


def generate_and_save_signal():
    """Generate ensemble signal using multiple timeframes."""
    try:
        # === Load model if exists ===
        model = None
        if MODEL_FILE.exists():
            try:
                model = joblib.load(MODEL_FILE)
                print("✅ Loaded existing model.")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")

        # === Load timeframe data ===
        data_sources = {}
        for label, file in {
            "daily": DAILY_FILE,
            "weekly": WEEKLY_FILE,
            "monthly": MONTHLY_FILE
        }.items():
            if file.exists():
                df = pd.read_csv(file)
                df = ensure_features(df)
                data_sources[label] = df
            else:
                print(f"⚠️ Missing file: {file}")

        if not data_sources:
            raise FileNotFoundError("No timeframe CSV files found")

        # === Ensemble weights ===
        weights = {"daily": 0.5, "weekly": 0.3, "monthly": 0.2}
        votes = {"BUY": 0.0, "SELL": 0.0}
        sub_signals = {}

        # === Compute trend-based signal per timeframe ===
        for tf, df in data_sources.items():
            sig = get_signal_from_trend(df)
            sub_signals[tf] = sig
            votes[sig] += weights[tf]

        # === Final ensemble signal ===
        final_signal = "BUY" if votes["BUY"] > votes["SELL"] else "SELL"
        confidence = round(abs(votes["BUY"] - votes["SELL"]) * 100, 2)

        # === Optional model refinement ===
        if model:
            features = {}
            for tf, df in data_sources.items():
                p = tf[0]  # d, w, m
                features.update({
                    f"ema21_{p}": df["ema21"].iloc[-1],
                    f"ema50_{p}": df["ema50"].iloc[-1],
                    f"atr14_{p}": df["atr14"].iloc[-1]
                })
            X = pd.DataFrame([features])
            try:
                pred = model.predict(X)[0]
                model_signal = "BUY" if pred == 1 else "SELL"
                if model_signal == final_signal:
                    confidence = min(100, confidence + 10)
                else:
                    confidence = max(0, confidence - 10)
            except Exception as e:
                print(f"⚠️ Model prediction skipped: {e}")

        # === Response ===
        response = {
            "final_signal": final_signal,
            "confidence": confidence,
            "sub_signals": sub_signals,
            "weights": weights,
            "timestamp": str(datetime.utcnow()),
        }

        # === Save files ===
        with open(SIGNALS_FILE, "w") as f:
            json.dump(response, f, indent=2)

        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.load(open(HISTORY_FILE))
            except Exception:
                history = []
        history.append(response)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)

        print(f"[{response['timestamp']}] ✅ Ensemble {final_signal} ({confidence}%) — {sub_signals}")
        return response

    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return {"error": str(e)}


def background_scheduler():
    while True:
        generate_and_save_signal()
        time.sleep(REFRESH_INTERVAL_SECS)


@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.utcnow())})


@app.route('/signal', methods=['GET'])
def signal_api():
    result = generate_and_save_signal()
    return jsonify(result)


@app.route('/history', methods=['GET'])
def history_api():
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
            return jsonify({"count": len(history), "history": history})
        except Exception as e:
            return jsonify({"error": f"Failed to read history: {e}"}), 500
    return jsonify({"message": "No signal history found"}), 404


if __name__ == '__main__':
    threading.Thread(target=background_scheduler, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
