import pandas as pd
import numpy as np
import os
import ta  # technical analysis library
import yfinance as yf
from loguru import logger

# === CONFIG ===
DATA_DIR = "data"
FILES = {
    "daily": "XAU_USD_Historical_Data_daily.csv",
    "weekly": "XAU_USD_Historical_Data_weekly.csv",
    "monthly": "XAU_USD_Historical_Data_monthly.csv",
    "hourly": "XAU_USD_Historical_Data_hourly.csv"
}
YF_SYMBOL = "GC=F"  # Gold futures symbol

# === Step 1: Auto-fetch if not present ===
def fetch_from_yahoo():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"📡 Fetching fresh data for {YF_SYMBOL} ...")

    intervals = {
        "daily": ("25y", "1d"),
        "weekly": ("25y", "1wk"),
        "monthly": ("25y", "1mo"),
        "hourly": ("2y", "1h")  # <= FIXED: limit hourly to 2 years
    }

    for tf, (period, interval) in intervals.items():
        try:
            df = yf.download(YF_SYMBOL, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"⚠️ {tf.capitalize()} data fetch failed for {YF_SYMBOL}.")
                continue

            df.reset_index(inplace=True)
            df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)

            # Ensure Close column
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df.rename(columns={"Adj Close": "Close"}, inplace=True)

            if "Close" not in df.columns:
                logger.warning(f"⚠️ No 'Close' column found in {tf} data. Columns: {df.columns}")

            file_path = os.path.join(DATA_DIR, FILES[tf])
            df.to_csv(file_path, index=False)
            logger.info(f"✅ Saved {tf} → {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"❌ Error fetching {tf} data: {e}")

# === Step 2: Load + clean + feature engineer ===
def load_and_prepare():
    """Load, clean, and feature-engineer multi-timeframe datasets."""
    datasets = {}

    # Auto-fetch missing files
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        logger.warning(f"Missing data files: {missing}. Fetching automatically...")
        fetch_from_yahoo()

    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"⚠️ File not found even after fetch: {path}")
            continue

        print(f"📥 Loading {tf.upper()} data from {path} ...")
        df = pd.read_csv(path)

        # --- Normalize Columns ---
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "price": "close",
            "last": "close",
            "timestamp": "date"
        })

        # --- Date conversion ---
        for col in ["date", "datetime", "time"]:
            if col in df.columns:
                df["date"] = pd.to_datetime(df[col], errors="coerce")
                break

        df = df.sort_values("date").dropna(subset=["date"])
        df.set_index("date", inplace=True)

        # --- Ensure close exists ---
        if "close" not in df.columns:
            logger.error(f"❌ Missing 'close' column in {fname}. Columns found: {df.columns}")
            continue

        # --- Technical Indicators ---
        try:
            df["return"] = df["close"].pct_change()
            df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
            df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
            df["ema_100"] = ta.trend.ema_indicator(df["close"], window=100)
            df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
            df["atr_14"] = ta.volatility.average_true_range(
                high=df["high"] if "high" in df.columns else df["close"],
                low=df["low"] if "low" in df.columns else df["close"],
                close=df["close"],
                window=14
            )
            df["volatility"] = df["close"].pct_change().rolling(10).std()
            df["momentum"] = ta.momentum.roc(df["close"], window=5)
        except Exception as e:
            logger.error(f"Error computing indicators for {tf}: {e}")
            continue

        # --- Drop incomplete rows ---
        df = df.dropna()

        # --- Add timeframe label ---
        df["timeframe"] = tf

        datasets[tf] = df
        print(f"✅ {tf.upper()} data ready → {len(df)} rows, features: {len(df.columns)}")

    return datasets


if __name__ == "__main__":
    data = load_and_prepare()
    for tf, df in data.items():
        print(f"\n📊 {tf.upper()} PREVIEW:")
        print(df.tail(3))
