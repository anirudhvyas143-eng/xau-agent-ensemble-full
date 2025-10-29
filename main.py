from flask import Flask, jsonify, request
import pandas as pd
import joblib
import json, threading, time, os, requests, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

app = Flask(__name__)

# === File paths ===
DAILY_FILE = Path("features_full_daily.csv")
WEEKLY_FILE = Path("features_full_weekly.csv")
MONTHLY_FILE = Path("features_full_monthly.csv")
MODEL_FILE = Path("model.pkl")
SIGNALS_FILE = Path("signals.json")
HISTORY_FILE = Path("signals_history.json")

REFRESH_INTERVAL_SECS = 86400  # 24 hours

# === Fetch and calculate indicators ===
def fetch_and_process_data():
    """Fetch XAU/USD data and compute EMA/ATR for daily, weekly, monthly."""
    print("📥 Fetching latest gold data...")
    symbol = "GC=F"  # Gold Futures (USD)
    df_daily = yf.download(symbol, period="1y", interval="1d")
    df_weekly = yf.download(symbol, period="2y", interval="1wk")
    df_monthly = yf.download(symbol, period="5y", interval="1mo")

    for df, label, file in [
        (df_daily, "daily", DAILY_FILE),
        (df_weekly, "weekly", WEEKLY_FILE),
        (df_monthly, "monthly", MONTHLY_FILE)
    ]:
        df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(window=14).mean()

        df = df.dropna()
        df.to_csv(file)
        print(f"✅ {label.capitalize()} data updated: {file.name}")

# === Train or load model ===
def train_or_load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    else:
        model = RandomForestClassifier()
        features = [
            "ema21_d", "ema50_d", "atr14_d",
            "ema21_w", "ema50_w", "atr14_w",
            "ema21_m", "ema50_m", "atr14_m"
        ]
        dummy_X = pd.DataFrame([[1]*9], columns=features)
        dummy_y = [1]
        model.fit(dummy_X, dummy_y)
        joblib.dump(model, MODEL_FILE)
        print("✅ Default model trained.")
        return model

# === Generate signal ===
def generate_and_save_signal():
    try:
        fetch_and_process_data()
        model = train_or_load_model()

        def load_latest(file):
            df = pd.read_csv(file)
            return df.iloc[-1]

        d, w, m = load_latest(DAILY_FILE), load_latest(WEEKLY_FILE), load_latest(MONTHLY_FILE)
        features = {
            "ema21_d": d["ema21"], "ema50_d": d["ema50"], "atr14_d": d["atr14"],
            "ema21_w": w["ema21"], "ema50_w": w["ema50"], "atr14_w": w["atr14"],
            "ema21_m": m["ema21"], "ema50_m": m["ema50"], "atr14_m": m["atr14"]
        }

        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][prediction] if hasattr(model, "predict_proba") else [0.0]
        signal = "BUY" if prediction == 1 else "SELL"

        response = {
            "signal": signal,
            "confidence": round(prob * 100, 2),
            "timestamp": str(datetime.now(timezone.utc)),
            "features_used": list(features.keys())
        }

        json.dump(response, open(SIGNALS_FILE, "w"), indent=2)

        history = json.load(open(HISTORY_FILE)) if HISTORY_FILE.exists() else []
        history.append(response)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)

        print(f"[{response['timestamp']}] ✅ {signal} ({response['confidence']}%) — saved and logged")
        return response

    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return {"error": str(e)}

# === Background auto-update ===
def background_scheduler():
    while True:
        generate_and_save_signal()
        try:
            # Self-ping to stay alive on Render
            requests.get("https://xau-agent-ensemble-full.onrender.com/")
        except Exception as e:
            print(f"Ping failed: {e}")
        time.sleep(REFRESH_INTERVAL_SECS)

# === Flask routes ===
@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})

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

@app.route('/dashboard')
def dashboard():
    import html
    try:
        latest_signal = json.load(open(SIGNALS_FILE))
    except Exception:
        latest_signal = {"signal": "N/A", "confidence": 0, "timestamp": "N/A"}

    signal = latest_signal.get("signal", "N/A")
    conf = latest_signal.get("confidence", 0)
    time_str = latest_signal.get("timestamp", "N/A")

    color = "#00C853" if signal == "BUY" else "#D50000" if signal == "SELL" else "#AAAAAA"

    html_content = f"""
    <html>
    <head>
        <title>XAU/USD Signal Dashboard</title>
        <meta http-equiv="refresh" content="300">
        <style>
            body {{
                background-color: #0d1117;
                color: #fff;
                font-family: 'Arial', sans-serif;
                text-align: center;
                padding-top: 40px;
            }}
            .signal-box {{
                display: inline-block;
                padding: 25px;
                border-radius: 15px;
                background-color: #161b22;
                box-shadow: 0 0 15px rgba(255,255,255,0.1);
                width: 400px;
            }}
            .signal {{
                font-size: 60px;
                font-weight: bold;
                color: {color};
            }}
            .confidence {{
                font-size: 22px;
                margin-top: 10px;
            }}
            .timestamp {{
                font-size: 14px;
                color: #bbb;
                margin-top: 10px;
            }}
            .button {{
                margin-top: 20px;
                padding: 10px 20px;
                font-size: 16px;
                color: #fff;
                background-color: #1f6feb;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }}
            .button:hover {{
                background-color: #2a7ffb;
            }}
            .footer {{
                margin-top: 30px;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="signal-box">
            <h2>XAU/USD Ensemble Signal</h2>
            <div class="signal">{html.escape(signal)}</div>
            <div class="confidence">Confidence: {conf}%</div>
            <div class="timestamp">Last Updated: {html.escape(time_str)}</div>
            <button class="button" onclick="fetch('/signal').then(() => window.location.reload())">
                🔄 Regenerate Signal
            </button>
        </div>
        <div class="footer">
            <p>Auto-updates every 24 hours • History logged</p>
        </div>
    </body>
    </html>
    """
    return html_content

# === Run ===
if __name__ == '__main__':
    threading.Thread(target=background_scheduler, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
