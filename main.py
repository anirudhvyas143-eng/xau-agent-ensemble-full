# main.py — XAU/USD AI Agent (AlphaVantage Only)
from flask import Flask, jsonify
import pandas as pd, numpy as np, joblib, json, threading, time, os, requests
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# === CONFIGURATION ===
# ======================================================
app = Flask(__name__)
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"
MODEL_DAY = ROOT / "model_day.pkl"
SCALER_DAY = ROOT / "scaler_day.pkl"
MODEL_HR = ROOT / "model_hr.pkl"
SCALER_HR = ROOT / "scaler_hr.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))
PORT = int(os.getenv("PORT", 10000))
SELF_PING_URL = os.getenv("SELF_PING_URL", None)
ALPHAV_API_KEY = "XWZFB7RP8I4SWCMZ"  # ✅ Your provided AlphaVantage API key

# ======================================================
# === INDICATORS ===
# ======================================================
def compute_indicators(df):
    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["mom5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()
    df.dropna(inplace=True)
    return df

# ======================================================
# === ALPHA VANTAGE FETCHERS ===
# ======================================================
def fetch_alpha_daily():
    print("📥 Fetching daily XAU/USD data (Alpha Vantage)...")
    url = (
        f"https://www.alphavantage.co/query?function=FX_DAILY"
        f"&from_symbol=XAU&to_symbol=USD&apikey={ALPHAV_API_KEY}&outputsize=full"
    )
    try:
        r = requests.get(url, timeout=20)
        data = r.json().get("Time Series FX (Daily)", {})
        if not data:
            raise ValueError("Empty daily dataset from Alpha Vantage.")
        df = pd.DataFrame(data).T
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.astype(float)
        df["Date"] = pd.to_datetime(df.index)
        df.sort_values("Date", inplace=True)
        df.to_csv(DAILY_FILE, index=False)
        print(f"✅ Saved daily data → {DAILY_FILE} ({len(df)} rows)")
        return df
    except Exception as e:
        print("❌ AlphaVantage daily fetch error:", e)
        if DAILY_FILE.exists():
            print("⚠️ Using cached daily data.")
            return pd.read_csv(DAILY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

def fetch_alpha_hourly():
    print("📥 Fetching hourly XAU/USD data (Alpha Vantage)...")
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY"
        f"&from_symbol=XAU&to_symbol=USD&interval=60min"
        f"&apikey={ALPHAV_API_KEY}&outputsize=full"
    )
    try:
        r = requests.get(url, timeout=20)
        data = r.json().get("Time Series FX (60min)", {})
        if not data:
            raise ValueError("Empty hourly dataset from Alpha Vantage.")
        df = pd.DataFrame(data).T
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.astype(float)
        df["Date"] = pd.to_datetime(df.index)
        df.sort_values("Date", inplace=True)
        df.to_csv(HOURLY_FILE, index=False)
        print(f"✅ Saved hourly data → {HOURLY_FILE} ({len(df)} rows)")
        return df
    except Exception as e:
        print("❌ AlphaVantage hourly fetch error:", e)
        if HOURLY_FILE.exists():
            print("⚠️ Using cached hourly data.")
            return pd.read_csv(HOURLY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

# ======================================================
# === MODELING + SIGNALS ===
# ======================================================
def train_model(X, y, model_path):
    if len(X) < 50:
        print("⚠️ Not enough samples to train model.")
        return None
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)
    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_path).replace(".pkl", "_scaler.pkl"))
    print(f"🤖 Model saved {model_path.name} (val acc={acc:.3f})")
    return model

def generate_signal(df, model_path, label):
    if df.empty:
        return {"label": label, "signal": "N/A", "confidence": 0}

    df = compute_indicators(df)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
    df.dropna(subset=features + ["target"], inplace=True)

    model = train_model(df[features], df["target"], model_path)
    if model is None:
        return {"label": label, "signal": "N/A", "confidence": 0}

    scaler = joblib.load(str(model_path).replace(".pkl", "_scaler.pkl"))
    last = df[features].iloc[[-1]]
    prob = float(model.predict_proba(scaler.transform(last))[0][1])
    pred = int(model.predict(scaler.transform(last))[0])
    signal = "BUY" if pred == 1 else "SELL"

    return {"label": label, "signal": signal, "confidence": round(prob * 100, 2)}

# ======================================================
# === PIPELINE ===
# ======================================================
def build_train_and_signal():
    daily_df = fetch_alpha_daily()
    hr_df = fetch_alpha_hourly()

    day_sig = generate_signal(daily_df, MODEL_DAY, "Daily")
    hr_sig = generate_signal(hr_df, MODEL_HR, "Hourly")

    combined = {
        "timestamp": str(datetime.now(timezone.utc)),
        "daily": day_sig,
        "hourly": hr_sig
    }

    json.dump(combined, open(SIGNALS_FILE, "w"), indent=2)
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
        except Exception:
            pass
    history.append(combined)
    json.dump(history[-100:], open(HISTORY_FILE, "w"), indent=2)

    print(f"[{combined['timestamp']}] 🕐 Hourly:{hr_sig['signal']}({hr_sig['confidence']}%) | 📅 Daily:{day_sig['signal']}({day_sig['confidence']}%)")
    return combined

# ======================================================
# === BACKGROUND REFRESH ===
# ======================================================
def background_loop():
    while True:
        try:
            build_train_and_signal()
            if SELF_PING_URL:
                requests.get(SELF_PING_URL, timeout=8)
        except Exception as e:
            print("Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)

# ======================================================
# === FLASK ROUTES ===
# ======================================================
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})

@app.route("/signal")
def signal_route():
    return jsonify(build_train_and_signal())

@app.route("/history")
def history_route():
    if HISTORY_FILE.exists():
        return jsonify(json.load(open(HISTORY_FILE)))
    return jsonify([])

@app.route("/dashboard")
def dashboard():
    try:
        current = json.load(open(SIGNALS_FILE))
    except Exception:
        current = {"daily": {"signal": "N/A"}, "hourly": {"signal": "N/A"}}

    color_day = "#0f0" if current["daily"]["signal"] == "BUY" else "#f55"
    color_hr = "#0f0" if current["hourly"]["signal"] == "BUY" else "#f55"

    return f"""
    <html><head><title>XAU/USD AI Dashboard</title></head>
    <body style="font-family:Arial;background:#0d1117;color:#fff;text-align:center;padding:40px;">
    <h1>XAU/USD Dual-Timeframe AI Agent (AlphaVantage)</h1>
    <div style="background:#161b22;padding:20px;border-radius:12px;">
      <h2>📅 Daily: <span style="color:{color_day}">{current['daily']['signal']}</span> ({current['daily'].get('confidence',0)}%)</h2>
      <h2>🕐 Hourly: <span style="color:{color_hr}">{current['hourly']['signal']}</span> ({current['hourly'].get('confidence',0)}%)</h2>
      <p><button onclick="fetch('/signal').then(()=>location.reload())">🔄 Refresh Signal</button></p>
      <p>Last Update: {current.get('timestamp','N/A')}</p>
    </div>
    </body></html>
    """

# ======================================================
# === START SERVER ===
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    print(f"🚀 Starting Flask on port {PORT} | Refresh every {REFRESH_INTERVAL_SECS}s (AlphaVantage only)")
    app.run(host="0.0.0.0", port=PORT)
