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
ALPHAV_API_KEY = "XWZFB7RP8I4SWCMZ"  # Your working Alpha Vantage key
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DAILY_FILE = os.path.join(DATA_DIR, "XAU_USD_Historical_Data_daily.csv")
HOURLY_FILE = os.path.join(DATA_DIR, "XAU_USD_Historical_Data_hourly.csv")

app = Flask(__name__)

# ======================================================
# 🪙 Fetch Alpha Vantage Data
# ======================================================

def fetch_alpha_daily():
    """Fetch daily gold price using Alpha Vantage DIGITAL_CURRENCY_DAILY endpoint."""
    print("📥 Fetching daily XAU/USD data (Alpha Vantage DIGITAL_CURRENCY_DAILY)...")

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": "XAU",
        "market": "USD",
        "apikey": ALPHAV_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "Time Series (Digital Currency Daily)" not in data:
            raise ValueError("Empty daily dataset from Alpha Vantage (XAU/USD).")

        df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient="index")
        df = df.rename(columns={
            "1a. open (USD)": "open",
            "2a. high (USD)": "high",
            "3a. low (USD)": "low",
            "4a. close (USD)": "close",
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


def fetch_alpha_hourly():
    """Simulate hourly gold price from latest daily data."""
    print("📥 Fetching simulated hourly XAU/USD data (from daily)...")

    try:
        if not os.path.exists(DAILY_FILE):
            raise FileNotFoundError("Daily file missing for hourly generation.")

        daily_df = pd.read_csv(DAILY_FILE)
        if len(daily_df) == 0:
            raise ValueError("Daily dataset empty.")

        # Use last close price and simulate 24 hourly fluctuations
        last_price = float(daily_df["close"].iloc[-1])
        hours = pd.date_range(end=datetime.utcnow(), periods=24, freq="H")
        hourly_data = {
            "timestamp": hours,
            "close": [last_price * (1 + (i - 12) / 1000) for i in range(24)]
        }

        hourly_df = pd.DataFrame(hourly_data)
        hourly_df.to_csv(HOURLY_FILE, index=False)
        print(f"✅ Simulated hourly data → {HOURLY_FILE} ({len(hourly_df)} rows)")
        return hourly_df

    except Exception as e:
        print(f"❌ AlphaVantage hourly fetch error: {e}")
        return pd.DataFrame()

# ======================================================
# 🤖 Train Model (Daily)
# ======================================================

def train_simple_model():
    """Train a simple model using daily gold price data."""
    if not os.path.exists(DAILY_FILE):
        print("⚠️ No daily file yet, skipping model training.")
        return None

    df = pd.read_csv(DAILY_FILE)
    if len(df) < 5:
        print("⚠️ Not enough data to train model.")
        return None

    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    X = df[["price"]].fillna(0)
    y = df["target"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    pickle.dump(model, open("model_day.pkl", "wb"))
    print(f"🤖 Model saved model_day.pkl (val acc={acc:.3f})")
    return model

# ======================================================
# 🌐 Flask Routes
# ======================================================

@app.route("/")
def home():
    daily_df = pd.read_csv(DAILY_FILE) if os.path.exists(DAILY_FILE) else pd.DataFrame()
    hourly_df = pd.read_csv(HOURLY_FILE) if os.path.exists(HOURLY_FILE) else pd.DataFrame()

    response = {
        "daily_latest": daily_df.tail(1).to_dict(orient="records"),
        "hourly_latest": hourly_df.tail(1).to_dict(orient="records"),
        "message": "AlphaVantage data fetched successfully"
    }
    return jsonify(response)


@app.route("/signal")
def signal():
    """Return the latest BUY/SELL signal from the trained model."""
    if not os.path.exists("model_day.pkl") or not os.path.exists(DAILY_FILE):
        return jsonify({"signal": "N/A", "reason": "Model or data missing"})

    model = pickle.load(open("model_day.pkl", "rb"))
    df = pd.read_csv(DAILY_FILE)
    latest_price = df["price"].iloc[-1]
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
    print("🚀 Starting Flask on port 10000 | Refresh every 900s (AlphaVantage only)")

    while True:
        daily = fetch_alpha_daily()
        hourly = fetch_alpha_hourly()
        train_simple_model()

        if not daily.empty:
            print(f"[{datetime.utcnow()}] 📅 Daily: {daily['price'].iloc[-1]:.2f}")
        if not hourly.empty:
            print(f"[{datetime.utcnow()}] 🕐 Hourly: {hourly['close'].iloc[-1]}")

        app.run(host="0.0.0.0", port=10000)
        print("⏳ Waiting 15 minutes before refreshing data...")
        time.sleep(900)
