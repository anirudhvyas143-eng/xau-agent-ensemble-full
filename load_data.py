import pandas as pd
import numpy as np
import os
import requests
from loguru import logger
from datetime import datetime

# === CONFIG ===
DATA_DIR = "data"
FILES = {
    "daily": "XAU_USD_Historical_Data_daily.csv",
    "weekly": "XAU_USD_Historical_Data_weekly.csv",
    "monthly": "XAU_USD_Historical_Data_monthly.csv",
    "hourly": "XAU_USD_Historical_Data_hourly.csv"
}

# --- Primary: RapidAPI (live snapshot) ---
RAPID_API_URL = "https://gold-price-live.p.rapidapi.com/get_metal_prices"
RAPID_API_HEADERS = {
    "x-rapidapi-host": "gold-price-live.p.rapidapi.com",
    "x-rapidapi-key": os.getenv("RAPID_API_KEY", "demo-key")
}

# --- Fallbacks ---
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "demo")
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
INVESTING_DAILY_URL = (
    "https://api.investing.com/api/financialdata/historical/8830?interval=P1D&pointscount=365"
)

# =====================================================
# 📡 STEP 1: Fetch data from available APIs
# =====================================================
def fetch_live_gold_data():
    """Fetch latest gold price using RapidAPI."""
    try:
        logger.info("📡 Fetching live gold price from RapidAPI...")
        response = requests.get(RAPID_API_URL, headers=RAPID_API_HEADERS, timeout=10)
        data = response.json()

        if "result" not in data or "gold" not in data["result"]:
            logger.warning("⚠️ Invalid RapidAPI response; skipping.")
            return None

        price = data["result"]["gold"].get("price")
        if not price:
            logger.warning("⚠️ No 'price' in RapidAPI result.")
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
        logger.error(f"❌ RapidAPI fetch failed: {e}")
        return None


def fetch_investing_daily():
    """Fetch daily gold data from Investing.com JSON API."""
    try:
        logger.info("📡 Fetching daily gold data from Investing.com ...")
        resp = requests.get(INVESTING_DAILY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        raw = resp.json().get("data", [])
        if not raw:
            raise ValueError("Empty Investing.com data")
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["rowDate"])
        df["close"] = df["last_close"]
        df["high"] = df["high"]
        df["low"] = df["low"]
        df = df[["date", "close", "high", "low"]].sort_values("date")
        logger.info(f"✅ Investing.com data fetched: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Investing.com fetch failed: {e}")
        return None


def fetch_alpha_hourly():
    """Fetch intraday (hourly) gold data from Alpha Vantage."""
    try:
        logger.info("📡 Fetching hourly gold data from Alpha Vantage ...")
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": "XAU",
            "to_symbol": "USD",
            "interval": "60min",
            "apikey": ALPHAVANTAGE_KEY,
            "outputsize": "compact"
        }
        r = requests.get(ALPHAVANTAGE_URL, params=params, timeout=10)
        data = r.json().get("Time Series FX (60min)", {})
        if not data:
            raise ValueError("Empty AlphaVantage data")
        df = pd.DataFrame(data).T
        df.columns = ["open", "high", "low", "close"]
        df = df.astype(float)
        df["date"] = pd.to_datetime(df.index)
        df = df.sort_values("date")
        logger.info(f"✅ AlphaVantage hourly data fetched: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"⚠️ AlphaVantage fetch failed: {e}")
        return None


# =====================================================
# 🧠 STEP 2: Load, clean, and normalize
# =====================================================
def load_and_prepare():
    """Load CSVs, fill missing via APIs, normalize, compute indicators."""
    datasets = {}
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Fill any missing CSVs ---
    missing_files = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing_files:
        logger.warning(f"⚠️ Missing files detected: {missing_files}")

        live_df = fetch_live_gold_data() or fetch_investing_daily() or fetch_alpha_hourly()
        if live_df is not None:
            for tf, fname in FILES.items():
                path = os.path.join(DATA_DIR, fname)
                live_df.to_csv(path, index=False)
                logger.info(f"🪙 Created fallback dataset: {path}")

    # --- Load and clean datasets ---
    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"⚠️ Still missing {tf}: {path}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {
            "price": "close",
            "last": "close",
            "timestamp": "date",
            "datetime": "date"
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        if "date" not in df.columns:
            df["date"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df = df.sort_values("date").set_index("date")

        for col in ["close", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"] if "close" in df.columns else np.nan

        try:
            df["return"] = df["close"].pct_change()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
            df["rsi_14"] = compute_rsi(df["close"], 14)
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
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
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
