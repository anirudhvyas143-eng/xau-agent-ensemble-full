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
    "monthly": "XAU_USD_Historical_Data_monthly.csv"
}
YF_SYMBOL = "GC=F"  # Gold futures symbol

# === Step 1: Auto-fetch if not present ===
def fetch_from_yahoo():
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info(f"Fetching fresh data for {YF_SYMBOL} ...")

    # ---- Daily data ----
    df_daily = yf.download(YF_SYMBOL, period="25y", interval="1d")
    if df_daily.empty:
        logger.warning("Daily data fetch failed.")
    else:
        df_daily.reset_index(inplace=True)
        df_daily.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
        if "Close" not in df_daily.columns and "Adj Close" in df_daily.columns:
            df_daily.rename(columns={"Adj Close": "Close"}, inplace=True)
        df_daily.to_csv(os.path.join(DATA_DIR, FILES["daily"]), index=False)
        logger.info(f"✅ Saved daily → {len(df_daily)} rows")

    # ---- Weekly data ----
    df_weekly = yf.download(YF_SYMBOL, period="25y", interval="1wk")
    if df_weekly.empty:
        logger.warning("Weekly data fetch failed.")
    else:
        df_weekly.reset_index(inplace=True)
        df_weekly.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
        if "Close" not in df_weekly.columns and "Adj Close" in df_weekly.columns:
            df_weekly.rename(columns={"Adj Close": "Close"}, inplace=True)
        df_weekly.to_csv(os.path.join(DATA_DIR, FILES["weekly"]), index=False)
        logger.info(f"✅ Saved weekly → {len(df_weekly)} rows")

    # ---- Monthly data ----
    df_monthly = yf.download(YF_SYMBOL, period="25y", interval="1mo")
    if df_monthly.empty:
        logger.warning("Monthly data fetch failed.")
    else:
        df_monthly.reset_index(inplace=True)
        df_monthly.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
        if "Close" not in df_monthly.columns and "Adj Close" in df_monthly.columns:
            df_monthly.rename(columns={"Adj Close": "Close"}, inplace=True)
        df_monthly.to_csv(os.path.join(DATA_DIR, FILES["monthly"]), index=False)
        logger.info(f"✅ Saved monthly → {len(df_monthly)} rows")

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
            print(f"⚠️ File not found even after fetch: {path}")
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
            raise ValueError(f"❌ Missing 'close' column in {fname}")

        # --- Technical Indicators ---
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
