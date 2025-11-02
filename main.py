import os
import pandas as pd
import requests
import time
from flask import Flask, jsonify
from datetime import datetime
import pickle
import threading
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

# ======================================================
# 🔧 CONFIGURATION
# ======================================================

# ✅ Only the three Alpha Vantage API keys you provided (no extras)
ALPHAV_API_KEYS = [
    "XWZFB7RP8I4SWCMZ",  # key A
    "XUU2PYO481XBYWR4",  # key B
    "94CMKYJJQUVN51AT",  # key C
]

# pick one at start (will rotate on failures)
ALPHAV_API_KEY = random.choice(ALPHAV_API_KEYS)
print(f"🔑 Starting with Alpha Vantage key ending with ...{ALPHAV_API_KEY[-4:]}")

SYMBOL = "GLD"  # GLD ETF used as reliable gold proxy (alpha supports symbol-based endpoints)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DAILY_FILE = os.path.join(DATA_DIR, f"{SYMBOL}_daily.csv")
HOURLY_FILE = os.path.join(DATA_DIR, f"{SYMBOL}_hourly.csv")

# 2 hours refresh to stay within free-tier rate limits
REFRESH_INTERVAL = 7200
print(f"⏱️ Refresh interval set to {REFRESH_INTERVAL // 60} minutes (safe for Alpha Vantage free tier).")

app = Flask(__name__)

# ======================================================
# 🔁 Helper: Rotate API keys if one fails
# ======================================================
def try_alpha_request(params):
    """Try AlphaVantage request with key rotation. Returns parsed JSON or {}."""
    global ALPHAV_API_KEY

    base = "https://www.alphavantage.co/query"
    for key in ALPHAV_API_KEYS:
        params["apikey"] = key
        try:
            r = requests.get(base, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"⚠️ Network/error with key ...{key[-4:]}: {e}")
            continue

        # If they returned an error/information about limits, skip to next
        msg = str(data)
        if "Thank you for using Alpha Vantage" in msg or "rate limit" in msg or "Invalid API call" in msg:
            print(f"⚠️ Key ending with ...{key[-4:]} returned info/limit message; rotating to next key.")
            continue

        # Successful-ish payload (we still validate presence of expected keys later)
        ALPHAV_API_KEY = key
        print(f"✅ Data returned using key ending with ...{key[-4:]}")
        return data

    print("❌ All Alpha Vantage keys exhausted or invalid for this request.")
    return {}

# ======================================================
# 🪙 Fetch Daily Data (TIME_SERIES_DAILY)
# ======================================================
def fetch_alpha_daily():
    print("📥 Fetching daily GLD data (Alpha Vantage TIME_SERIES_DAILY)...")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "outputsize": "compact"
    }

    try:
        data = try_alpha_request(params)
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Empty/invalid daily dataset from Alpha Vantage: {data}")

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.to_csv(DAILY_FILE)
        print(f"✅ Saved daily data → {DAILY_FILE} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"❌ AlphaVantage daily fetch error: {e}")
        return pd.DataFrame()

# ======================================================
# ⏰ Fetch Hourly Data (TIME_SERIES_INTRADAY)
# ======================================================
def fetch_alpha_hourly():
    print("📥 Fetching hourly GLD data (Alpha Vantage TIME_SERIES_INTRADAY)...")
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": SYMBOL,
        "interval": "60min",
        "outputsize": "compact"
    }

    try:
        data = try_alpha_request(params)
        if "Time Series (60min)" not in data:
            raise ValueError(f"Empty/invalid hourly dataset from Alpha Vantage: {data}")

        df = pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.to_csv(HOURLY_FILE)
        print(f"✅ Saved hourly data → {HOURLY_FILE} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"❌ AlphaVantage hourly fetch error: {e}")
        return pd.DataFrame()

# ======================================================
# 🧠 Train Model
# ======================================================
def train_model():
    if not os.path.exists(DAILY_FILE):
        print("⚠️ No daily file yet, skipping model training.")
        return

    df = pd.read_csv(DAILY_FILE)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved.")

# ======================================================
# 🔁 Background Task
# ======================================================
def background_task():
    while True:
        daily_df = fetch_alpha_daily()
        hourly_df = fetch_alpha_hourly()
        if not daily_df.empty:
            train_model()
        print(f"⏳ Waiting {REFRESH_INTERVAL // 60} minutes before refreshing data...\n")
        time.sleep(REFRESH_INTERVAL)

# ======================================================
# 🌐 Flask Routes
# ======================================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "time": datetime.utcnow().isoformat(),
        "active_api_key": ALPHAV_API_KEY[-4:],
        "message": "XAU Agent (Alpha Vantage multi-key rotation) is live 🚀"
    })

@app.route("/predict")
def predict():
    if not os.path.exists("model.pkl"):
        return jsonify({"error": "Model not trained yet"})
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(DAILY_FILE).tail(1)
    X_latest = df[["open", "high", "low", "close", "volume"]]
    pred = model.predict(X_latest)[0]
    return jsonify({"prediction": int(pred)})

# ======================================================
# 🚀 Main Entrypoint
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=background_task, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
