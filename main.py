import os
import pandas as pd
import requests
import time
from flask import Flask, jsonify
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================================================
# 🔧 CONFIGURATION
# ======================================================
ALPHAV_API_KEY = "XUU2PYO481XBYWR4"   # ✅ Working Alpha Vantage key
RAPID_API_KEY = os.getenv("RAPID_API_KEY", "58cdeafeb6msh0a464937e4dacfep1554e1jsned5c2fa7d90e")  # 🔑 RapidAPI key (set via environment)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DAILY_FILE = os.path.join(DATA_DIR, "XAU_USD_Historical_Data_daily.csv")
HOURLY_FILE = os.path.join(DATA_DIR, "XAU_USD_Historical_Data_hourly.csv")

app = Flask(__name__)

# ======================================================
# 🪙 Alpha Vantage — Daily Commodity Data
# ======================================================
def fetch_alpha_daily():
    print("📥 Fetching daily XAU/USD data (Alpha Vantage COMMODITY_DAILY)...")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "COMMODITY_DAILY",
        "symbol": "XAUUSD",
        "apikey": ALPHAV_API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "data" not in data or len(data["data"]) == 0:
            raise ValueError("Empty daily dataset from Alpha Vantage COMMODITY_DAILY.")

        df = pd.DataFrame(data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
        df = df.sort_values("timestamp")
        df.to_csv(DAILY_FILE, index=False)

        print(f"✅ Saved daily data → {DAILY_FILE} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"❌ Alpha Vantage daily fetch error: {e}")
        return pd.DataFrame()

# ======================================================
# ⏱ RapidAPI — Hourly Gold Data
# ======================================================
def fetch_hourly_rapidapi():
    print("📥 Fetching hourly XAU/USD data (RapidAPI Gold-Price-Live)...")
    url = "https://gold-price-live.p.rapidapi.com/get_metal_prices/XAUUSD"
    headers = {
        "x-rapidapi-key": RAPID_API_KEY.strip(),
        "x-rapidapi-host": "gold-price-live.p.rapidapi.com"
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        data = r.json()

        if "result" not in data or not data["result"]:
            raise ValueError("Empty hourly dataset from RapidAPI.")

        df = pd.DataFrame(data["result"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low", "price": "close"
        })
        df.to_csv(HOURLY_FILE, index=False)
        print(f"✅ Saved hourly data → {HOURLY_FILE} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"❌ RapidAPI hourly fetch error: {e}")
        return pd.DataFrame()

# ======================================================
# 🤖 Train Model (Daily Close)
# ======================================================
def train_simple_model():
    if not os.path.exists(DAILY_FILE):
        print("⚠️ No daily file yet, skipping model training.")
        return None

    df = pd.read_csv(DAILY_FILE)
    if "close" not in df.columns or len(df) < 10:
        print("⚠️ Not enough valid data for model.")
        return None

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["close"]].fillna(0)
    y = df["target"].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    pickle.dump(model, open("model_day.pkl", "wb"))
    print(f"🤖 Model saved (val acc = {acc:.3f})")
    return model

# ======================================================
# 🌐 Flask Endpoints
# ======================================================
@app.route("/")
def home():
    daily_df = pd.read_csv(DAILY_FILE) if os.path.exists(DAILY_FILE) else pd.DataFrame()
    hourly_df = pd.read_csv(HOURLY_FILE) if os.path.exists(HOURLY_FILE) else pd.DataFrame()
    return jsonify({
        "daily_latest": daily_df.tail(1).to_dict(orient="records"),
        "hourly_latest": hourly_df.tail(1).to_dict(orient="records"),
        "message": "AlphaVantage (daily) + RapidAPI (hourly)"
    })

@app.route("/signal")
def signal():
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
# 🚀 MAIN LOOP
# ======================================================
if __name__ == "__main__":
    print("🚀 Starting Flask on port 10000 | Refresh every 900s (AlphaVantage + RapidAPI)")
    while True:
        daily = fetch_alpha_daily()
        hourly = fetch_hourly_rapidapi()
        train_simple_model()

        if not daily.empty:
            print(f"[{datetime.utcnow()}] 📅 Daily close: {daily['close'].iloc[-1]}")
        if not hourly.empty:
            print(f"[{datetime.utcnow()}] 🕐 Hourly close: {hourly['close'].iloc[-1]}")

        app.run(host="0.0.0.0", port=10000)
        print("⏳ Waiting 15 minutes before refreshing data...")
        time.sleep(900)
