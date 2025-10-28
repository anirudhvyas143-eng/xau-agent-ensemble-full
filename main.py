from flask import Flask, jsonify, request
import pandas as pd
import joblib
import json, threading, time, os
from datetime import datetime, timezone
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# === File paths ===
DAILY_FILE = Path("features_full_daily.csv")
WEEKLY_FILE = Path("features_full_weekly.csv")
MONTHLY_FILE = Path("features_full_monthly.csv")
MODEL_FILE = Path("model.pkl")
SIGNALS_FILE = Path("signals.json")
HISTORY_FILE = Path("signals_history.json")

# Run signal generation every 15 minutes
REFRESH_INTERVAL_SECS = 900


def train_default_model():
    """Train a fallback dummy model using all expected features (9 total)."""
    model = RandomForestClassifier()
    dummy_features = [
        "ema21_d", "ema50_d", "atr14_d",
        "ema21_w", "ema50_w", "atr14_w",
        "ema21_m", "ema50_m", "atr14_m"
    ]
    dummy_X = pd.DataFrame([[1]*9], columns=dummy_features)
    dummy_y = [1]
    model.fit(dummy_X, dummy_y)
    joblib.dump(model, MODEL_FILE)
    print("✅ Default model (9 features) trained.")
    return model


def generate_and_save_signal():
    """Generate multi-timeframe ensemble signal and save."""
    try:
        if not MODEL_FILE.exists():
            print("⚙️ Training default model (model.pkl not found)...")
            model = train_default_model()
        else:
            model = joblib.load(MODEL_FILE)

        # === Load available timeframe data ===
        data_sources = {}
        for label, file in {
            "daily": DAILY_FILE,
            "weekly": WEEKLY_FILE,
            "monthly": MONTHLY_FILE
        }.items():
            if file.exists():
                df = pd.read_csv(file)
                data_sources[label] = df
            else:
                print(f"⚠️ Missing file: {file}")

        if not data_sources:
            raise FileNotFoundError("No timeframe CSV files found")

        # === Extract latest feature values ===
        def get_latest_features(df, prefix):
            return {
                f"ema21_{prefix}": df["ema21"].iloc[-1],
                f"ema50_{prefix}": df["ema50"].iloc[-1],
                f"atr14_{prefix}": df["atr14"].iloc[-1],
            }

        features = {}
        for timeframe, df in data_sources.items():
            features.update(get_latest_features(df, timeframe[0]))  # d, w, m suffixes

        X = pd.DataFrame([features])

        # === Predict ===
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][prediction] if hasattr(model, "predict_proba") else 0.0
        signal = "BUY" if prediction == 1 else "SELL"

        response = {
            "signal": signal,
            "confidence": round(prob * 100, 2),
            "timestamp": str(datetime.now(timezone.utc)),
            "features_used": list(features.keys()),
        }

        # === Save signal ===
        with open(SIGNALS_FILE, "w") as f:
            json.dump(response, f, indent=2)

        # === Append to history ===
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.load(open(HISTORY_FILE))
            except Exception:
                history = []
        history.append(response)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)

        print(f"[{response['timestamp']}] ✅ {signal} ({response['confidence']}%) — saved and logged")
        return response

    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return {"error": str(e)}


def background_scheduler():
    """Auto-update signal every 15 min."""
    while True:
        generate_and_save_signal()
        time.sleep(REFRESH_INTERVAL_SECS)


# === Combine multiple timeframe signals ===
def get_ensemble_signal(daily, weekly, monthly):
    signals = [daily, weekly, monthly]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count > sell_count:
        return "BUY"
    elif sell_count > buy_count:
        return "SELL"
    else:
        return "NEUTRAL"


# === Routes ===

@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})


@app.route('/signal', methods=['GET'])
def signal_api():
    """Manually trigger signal generation."""
    result = generate_and_save_signal()
    return jsonify(result)


@app.route('/history', methods=['GET'])
def history_api():
    """Retrieve full signal history."""
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
            return jsonify({"count": len(history), "history": history})
        except Exception as e:
            return jsonify({"error": f"Failed to read history: {e}"}), 500
    return jsonify({"message": "No signal history found"}), 404


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive JSON input with daily/weekly/monthly data, return ensemble prediction.
    Example input:
    {
      "daily": {"ema21": 2350, "ema50": 2348, "atr14": 14.2},
      "weekly": {"ema21": 2360, "ema50": 2350, "atr14": 32.1},
      "monthly": {"ema21": 2380, "ema50": 2330, "atr14": 75.0}
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else train_default_model()

        signals = {}
        for tf in ["daily", "weekly", "monthly"]:
            if tf in data:
                df = pd.DataFrame([data[tf]])
                pred = model.predict(df)[0]
                signals[tf] = "BUY" if pred == 1 else "SELL"
            else:
                signals[tf] = "N/A"

        ensemble = get_ensemble_signal(
            signals.get("daily"), signals.get("weekly"), signals.get("monthly")
        )

        result = {
            "status": "success",
            "signals": signals,
            "ensemble": ensemble,
            "timestamp": str(datetime.now(timezone.utc))
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Simple web dashboard for signal visualization."""
    import html

    latest_signal = {}
    try:
        if SIGNALS_FILE.exists():
            latest_signal = json.load(open(SIGNALS_FILE))
    except Exception as e:
        latest_signal = {"error": str(e)}

    history_count = 0
    if HISTORY_FILE.exists():
        try:
            hist = json.load(open(HISTORY_FILE))
            history_count = len(hist)
        except Exception:
            pass

    signal = latest_signal.get("signal", "N/A")
    conf = latest_signal.get("confidence", 0)
    time_str = latest_signal.get("timestamp", "N/A")

    color = "#00C853" if signal == "BUY" else "#D50000" if signal == "SELL" else "#AAAAAA"

    html_content = f"""
    <html>
    <head>
        <title>XAU-USD Signal Dashboard</title>
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
            <p>History entries: {history_count}</p>
            <p>Auto-refreshes every 5 minutes</p>
        </div>
    </body>
    </html>
    """
    return html_content


if __name__ == '__main__':
    threading.Thread(target=background_scheduler, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
    
