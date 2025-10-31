import pandas as pd
import numpy as np
import os
import requests
from loguru import logger

# === CONFIG ===
DATA_DIR = "data"
FILES = {
    "daily": "XAU_USD_Historical_Data_daily.csv",
    "weekly": "XAU_USD_Historical_Data_weekly.csv",
    "monthly": "XAU_USD_Historical_Data_monthly.csv",
    "hourly": "XAU_USD_Historical_Data_hourly.csv"
}

RAPID_API_URL = "https://gold-price-live.p.rapidapi.com/get_metal_prices"
RAPID_API_HEADERS = {
    "x-rapidapi-host": "gold-price-live.p.rapidapi.com",
    "x-rapidapi-key": os.getenv("RAPID_API_KEY", "demo-key")
}


# =====================================================
# 📡 STEP 1: Fetch gold data from API (or fallback)
# =====================================================
def fetch_live_gold_data():
    """Fetch latest gold price data using RapidAPI or fallback."""
    try:
        logger.info("📡 Fetching latest gold price data from RapidAPI...")
        response = requests.get(RAPID_API_URL, headers=RAPID_API_HEADERS, timeout=10)
        data = response.json()

        if "result" not in data:
            logger.warning("⚠️ Unexpected API response — fallback to last CSV available.")
            return None

        prices = data["result"].get("gold", {})
        price = prices.get("price", None)
        if price is None:
            logger.warning("⚠️ No gold price field found in API data.")
            return None

        df = pd.DataFrame({
            "date": [pd.Timestamp.utcnow()],
            "close": [price],
            "high": [price * 1.001],
            "low": [price * 0.999]
        })
        logger.info(f"✅ Live gold price fetched: {price}")
        return df

    except Exception as e:
        logger.error(f"❌ Live gold fetch failed: {e}")
        return None


# =====================================================
# 🧠 STEP 2: Load, clean, and normalize data
# =====================================================
def load_and_prepare():
    """Load available CSVs, clean columns, and return dictionary of DataFrames."""
    datasets = {}

    os.makedirs(DATA_DIR, exist_ok=True)
    missing_files = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]

    # === Attempt to fill missing files with live data ===
    if missing_files:
        logger.warning(f"⚠️ Missing files detected: {missing_files}")
        live_df = fetch_live_gold_data()
        if live_df is not None:
            for tf, fname in FILES.items():
                path = os.path.join(DATA_DIR, fname)
                live_df.to_csv(path, index=False)
                logger.info(f"🪙 Created fallback data → {path}")

    # === Load available datasets ===
    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"⚠️ Still missing {tf} data: {path}")
            continue

        logger.info(f"📥 Loading {tf.upper()} data from {path}")
        df = pd.read_csv(path)

        # --- Normalize columns ---
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {
            "price": "close",
            "last": "close",
            "timestamp": "date",
            "datetime": "date"
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # --- Date handling ---
        if "date" not in df.columns:
            df["date"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df.dropna(subset=["date"], inplace=True)
        df = df.sort_values("date")
        df.set_index("date", inplace=True)

        # --- Ensure price columns exist ---
        for col in ["close", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"] if "close" in df.columns else np.nan

        # --- Basic derived metrics (simple, no ta lib needed) ---
        try:
            df["return"] = df["close"].pct_change()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
            df["rsi_14"] = compute_rsi(df["close"], window=14)
            df["atr_14"] = compute_atr(df)
            df["volatility"] = df["close"].pct_change().rolling(10).std()
            df["momentum"] = df["close"] - df["close"].shift(10)
        except Exception as e:
            logger.error(f"⚠️ Indicator calc failed for {tf}: {e}")
            continue

        df.dropna(inplace=True)
        df["timeframe"] = tf
        datasets[tf] = df
        logger.info(f"✅ {tf.upper()} ready → {len(df)} rows")

    return datasets


# =====================================================
# 🧮 SUPPORTING FUNCTIONS
# =====================================================
def compute_rsi(series, window=14):
    """Simplified RSI calculation."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df, window=14):
    """Simplified Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# =====================================================
# 🧾 RUN DIRECTLY
# =====================================================
if __name__ == "__main__":
    data = load_and_prepare()
    for tf, df in data.items():
        print(f"\n📊 {tf.upper()} PREVIEW:")
        print(df.tail(3))
