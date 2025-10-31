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
MODEL_HR = ROOT / "model_hr.pkl"
SCALER_DAY = ROOT / "scaler_day.pkl"
SCALER_HR = ROOT / "scaler_hr.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))
PORT = int(os.getenv("PORT", 10000))
SELF_PING_URL = os.getenv("SELF_PING_URL", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "demo")
ALPHAV_API_KEY = os.getenv("ALPHAV_API_KEY", "demo")

# ======================================================
# === INDICATOR ENGINE ===
# ======================================================
def compute_indicators(df):
    """Compute core indicators used in both models."""
    df = df.copy()
    df["ema8"] = df["Close"].ewm(span=8).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["mom5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()

    return df.dropna().reset_index(drop=True)

# ======================================================
# === DATA FETCHERS ===
# ======================================================
def fetch_investing_daily():
    """Fetch XAU/USD daily data via RapidAPI."""
    url = "https://investing-com.p.rapidapi.com/price/historical"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "investing-com.p.rapidapi.com"
    }
    params = {"symbol": "XAU/USD", "interval": "1d", "from": "2000-01-01"}

    try:
        print("📥 Fetching daily data...")
        r = requests.get(url, headers=headers, params=params, timeout=20)
        data = r.json()
        df = pd.DataFrame(data["data"])
        df["Date"] = pd.to_datetime(df["date"])
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        df = df[["Date", "Open", "High", "Low", "Close"]].sort_values("Date")
        df.to_csv(DAILY_FILE, index=False)
        return df
    except Exception as e:
        print("❌ RapidAPI error:", e)
        if DAILY_FILE.exists():
            print("⚠️ Using cached daily data.")
            return pd.read_csv(DAILY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

def fetch_alpha_hourly():
    """Fetch hourly XAU/USD data via AlphaVantage."""
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY"
        f"&from_symbol=XAU&to_symbol=USD&interval=60min"
        f"&apikey={ALPHAV_API_KEY}&outputsize=full"
    )
    try:
        print("📥 Fetching hourly data...")
        r = requests.get(url, timeout=20)
        data = r.json().get("Time Series FX (60min)", {})
        if not data:
            raise ValueError("Empty response from AlphaVantage.")
        df = pd.DataFrame(data).T
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.astype(float)
        df["Date"] = pd.to_datetime(df.index)
        df = df.sort_values("Date")
        df.to_csv(HOURLY_FILE, index=False)
        return df
    except Exception as e:
        print("❌ AlphaVantage error:", e)
        if HOURLY_FILE.exists():
            print("⚠️ Using cached hourly data.")
            return pd.read_csv(HOURLY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

# ======================================================
# === MODEL TRAINING + SIGNAL GENERATION ===
# ======================================================
def train_model(X, y, model_path, scaler_path):
    """Train RandomForest model."""
    if len(X) < 50:
        print("⚠️ Not enough data to train model.")
        return None, None

    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"🤖 Model saved → {model_path.name} (val acc={acc:.3f})")
    return model, scaler

def generate_signal(df, model_path, scaler_path, label):
    df = compute_indicators(df)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
    df = df.dropna(subset=features + ["target"])
    if df.empty:
        return {"label": label, "signal": "N/A", "confidence": 0}

    model, scaler = train_model(df[features], df["target"], model_path, scaler_path)
    if not model or not scaler:
        return {"label": label, "signal": "N/A", "confidence": 0}

    last = df[features].iloc[[-1]]
    pred = model.predict(scaler.transform(last))[0]
    prob = model.predict_proba(scaler.transform(last))[0][1]
    signal = "BUY" if pred == 1 else "SELL"
    return {
        "label": label,
        "timestamp": str(datetime.now(timezone.utc)),
        "signal": signal,
        "confidence": round(prob * 100, 2)
    }

# ======================================================
# === FULL PIPELINE ===
# ======================================================
def build_train_and_signal():
    daily = fetch_investing_daily()
    hourly = fetch_alpha_hourly()
    daily_sig = generate_signal(daily, MODEL_DAY, SCALER_DAY, "Daily")
    hourly_sig = generate_signal(hourly, MODEL_HR, SCALER_HR, "Hourly")

    combined = {
        "timestamp": str(datetime.now(timezone.utc)),
        "daily": daily_sig,
        "hourly": hourly_sig
    }

    json.dump(combined, open(SIGNALS_FILE, "w"), indent=2)
    hist = []
    if HISTORY_FILE.exists():
        try:
            hist = json.load(open(HISTORY_FILE))
        except Exception:
            pass
    hist.append(combined)
    json.dump(hist[-100:], open(HISTORY_FILE, "w"), indent=2)

    print(f"[{combined['timestamp']}] 📊 Daily={daily_sig['signal']}({daily_sig['confidence']}%) | Hourly={hourly_sig['signal']}({hourly_sig['confidence']}%)")
    return combined

# ======================================================
# === BACKGROUND REFRESH ===
# ======================================================
def background_loop():
    while True:
        try:
            build_train_and_signal()
            if SELF_PING_URL:
                try:
                    requests.get(SELF_PING_URL, timeout=5)
                except Exception:
                    pass
        except Exception as e:
            print("⚠️ Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)

# ======================================================
# === FLASK ROUTES ===
# ======================================================
@app.route("/")
def home():
    return jsonify({"status": "running", "time": str(datetime.now(timezone.utc))})

@app.route("/signal")
def signal_now():
    return jsonify(build_train_and_signal())

@app.route("/history")
def get_history():
    if HISTORY_FILE.exists():
        try:
            return jsonify(json.load(open(HISTORY_FILE)))
        except Exception:
            return jsonify([])
    return jsonify([])

@app.route("/dashboard")
def dashboard():
    try:
        data = json.load(open(SIGNALS_FILE))
    except Exception:
        data = {"daily": {"signal": "N/A", "confidence": 0}, "hourly": {"signal": "N/A", "confidence": 0}}
    color_day = "#00ff7f" if data["daily"]["signal"] == "BUY" else "#ff4d4d"
    color_hr = "#00ff7f" if data["hourly"]["signal"] == "BUY" else "#ff4d4d"
    return f"""
    <html><head><title>XAU/USD AI Dashboard</title><meta http-equiv="refresh" content="600"></head>
    <body style="font-family:Arial;background:#0d1117;color:#fff;text-align:center;padding:40px;">
      <h1>🪙 XAU/USD Dual-Timeframe AI Agent</h1>
      <div style="background:#161b22;padding:20px;border-radius:12px;">
        <h2>📅 Daily: <span style="color:{color_day}">{data['daily']['signal']}</span> ({data['daily']['confidence']}%)</h2>
        <h2>🕐 Hourly: <span style="color:{color_hr}">{data['hourly']['signal']}</span> ({data['hourly']['confidence']}%)</h2>
        <p>Last Update: {data.get('timestamp','N/A')}</p>
        <p><button onclick="fetch('/signal').then(()=>location.reload())">🔄 Refresh</button></p>
      </div>
    </body></html>
    """

# ======================================================
# === MAIN ENTRYPOINT ===
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    print(f"🚀 Flask app running on port {PORT} | Refresh every {REFRESH_INTERVAL_SECS}s")
    app.run(host="0.0.0.0", port=PORT)
