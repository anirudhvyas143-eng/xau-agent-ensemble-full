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

# ======================================================
# 🔧 CONFIGURATION
# ======================================================
ALPHAV_API_KEY = "XUU2PYO481XBYWR4"  # Your working Alpha Vantage key
SYMBOL = "GLD"  # GLD ETF tracks Gold price closely

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DAILY_FILE = os.path.join(DATA_DIR, f"{SYMBOL}_daily.csv")
HOURLY_FILE = os.path.join(DATA_DIR, f"{SYMBOL}_hourly.csv")

REFRESH_INTERVAL = 900  # 15 minutes

app = Flask(__name__)

# ======================================================
# 🪙 Fetch Daily Data (Alpha Vantage TIME_SERIES_DAILY)
# ======================================================
def fetch_alpha_daily():
    """Fetch daily GLD data from Alpha Vantage (TIME_SERIES_DAILY)."""
    print("📥 Fetching daily GLD data (Alpha Vantage TIME_SERIES_DAILY)...")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "apikey": ALPHAV_API_KEY,
        "outputsize": "compact"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "Time Series (Daily)" not in data:
            raise ValueError(f"Empty dataset from Alpha Vantage: {data}")

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
# ⏰ Fetch Hourly Data (Alpha Vantage TIME_SERIES_INTRADAY)
# ======================================================
def fetch_alpha_hourly():
    """Fetch hourly GLD data using Alpha Vantage TIME_SERIES_INTRADAY."""
    print("📥 Fetching hourly GLD data (Alpha Vantage TIME_SERIES_INTRADAY)...")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": SYMBOL,
        "interval": "60min",
        "apikey": ALPHAV_API_KEY,
        "outputsize": "compact"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "Time Series (60min)" not in data:
            raise ValueError(f"Empty hourly dataset from Alpha Vantage: {data}")

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
# 🤖 Train Model (Daily)
# ======================================================
def train_simple_model():
    """Train a simple RandomForest model using daily close prices."""
    if not os.path.exists(DAILY_FILE):
        print("⚠️ No daily file yet, skipping model training.")
        return None

    df = pd.read_csv(DAILY_FILE)
    if "close" not in df.columns or len(df) < 5:
        print("⚠️ Not enough valid data to train model.")
        return None

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["close"]].fillna(0)
    y = df["target"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    pickle.dump(model, open("model_day.pkl", "wb"))
    print(f"🤖 Model trained and saved → model_day.pkl (val acc={acc:.3f})")
    return model


# ======================================================
# 🔁 Background Data Loop
# ======================================================
def data_refresh_loop():
    while True:
        daily = fetch_alpha_daily()
        hourly = fetch_alpha_hourly()
        train_simple_model()

        if not daily.empty:
            print(f"[{datetime.utcnow()}] 📅 Daily Close: {daily['close'].iloc[-1]}")
        if not hourly.empty:
            print(f"[{datetime.utcnow()}] 🕐 Hourly Close: {hourly['close'].iloc[-1]}")

        print("⏳ Waiting 15 minutes before refreshing data...\n")
        time.sleep(REFRESH_INTERVAL)


# ======================================================
# 🌐 Flask Endpoints
# ======================================================
@app.route("/")
def home():
    daily_df = pd.read_csv(DAILY_FILE) if os.path.exists(DAILY_FILE) else pd.DataFrame()
    hourly_df = pd.read_csv(HOURLY_FILE) if os.path.exists(HOURLY_FILE) else pd.DataFrame()

    response = {
        "daily_latest": daily_df.tail(1).to_dict(orient="records"),
        "hourly_latest": hourly_df.tail(1).to_dict(orient="records"),
        "message": "✅ AlphaVantage GLD (Gold proxy) data fetched successfully"
    }
    return jsonify(response)


@app.route("/signal")
def signal():
    """Return BUY/SELL signal from model based on latest close price."""
    if not os.path.exists("model_day.pkl") or not os.path.exists(DAILY_FILE):
        return jsonify({"signal": "N/A", "reason": "Model or data missing"})

    model = pickle.load(open("model_day.pkl", "rb"))
    df = pd.read_csv(DAILY_FILE)
    latest_price = df["close"].iloc[-1]
    pred = model.predict([[latest_price]])[0]
    signal = "BUY" if pred == 1 else "SELL"

    return jsonify({
        "timestamp": str(datetime.utcnow()),
        "latest_price": latest_price,
        "signal": signal
    })


# ======================================================
# 🚀 MAIN ENTRY POINT
# ======================================================
if __name__ == "__main__":
    print("🚀 Starting Flask on port 10000 | Refresh every 900s (AlphaVantage only)")
    t = threading.Thread(target=data_refresh_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=10000)
