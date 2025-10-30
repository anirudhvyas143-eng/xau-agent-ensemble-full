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

# =====================================================
# 📡 STEP 1: Auto-fetch data if not present
# =====================================================
def fetch_from_yahoo():
    """Download and cache multi-timeframe gold data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"📡 Fetching fresh data for {YF_SYMBOL} ...")

    intervals = {
        "daily": ("25y", "1d"),
        "weekly": ("25y", "1wk"),
        "monthly": ("25y", "1mo"),
        "hourly": ("2y", "1h")  # ✅ FIXED: hourly restricted to 2 years
    }

    for tf, (period, interval) in intervals.items():
        try:
            logger.info(f"⬇️ Fetching {tf} data ({period}, {interval})...")
            df = yf.download(YF_SYMBOL, period=period, interval=interval, progress=False)

            if df.empty:
                logger.warning(f"⚠️ {tf.capitalize()} data fetch failed for {YF_SYMBOL}.")
                continue

            df.reset_index(inplace=True)
            df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)

            # ✅ Ensure Close column presence
            if "Close" not in df.columns:
                if "Adj Close" in df.columns:
                    df.rename(columns={"Adj Close": "Close"}, inplace=True)
                else:
                    df["Close"] = df.iloc[:, 0]  # fallback to first numeric column

            file_path = os.path.join(DATA_DIR, FILES[tf])
            df.to_csv(file_path, index=False)
            logger.info(f"✅ Saved {tf} → {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"❌ Error fetching {tf} data: {e}")


# =====================================================
# 🧠 STEP 2: Load, clean, and feature engineer
# =====================================================
def load_and_prepare():
    """Load, clean, and compute basic technical indicators for multi-timeframes."""
    datasets = {}

    # --- Auto-fetch if missing ---
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        logger.warning(f"⚠️ Missing data files: {missing}. Fetching automatically...")
        fetch_from_yahoo()

    # --- Load each timeframe ---
    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"⚠️ File not found even after fetch: {path}")
            continue

        print(f"📥 Loading {tf.upper()} data from {path} ...")
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

        # --- Date conversion ---
        for col in ["date", "time"]:
            if col in df.columns:
                df["date"] = pd.to_datetime(df[col], errors="coerce")
                break

        df = df.sort_values("date").dropna(subset=["date"])
        df.set_index("date", inplace=True)

        # --- Ensure 'close' exists ---
        if "close" not in df.columns:
            logger.error(f"❌ Missing 'close' column in {fname}. Columns found: {df.columns}")
            continue

        # --- Compute base technicals ---
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
            logger.error(f"⚠️ Error computing indicators for {tf}: {e}")
            continue

        # --- Clean ---
        df.dropna(inplace=True)
        df["timeframe"] = tf

        datasets[tf] = df
        print(f"✅ {tf.upper()} ready → {len(df)} rows | {len(df.columns)} features")

    return datasets


# =====================================================
# 🧾 RUN DIRECTLY
# =====================================================
if __name__ == "__main__":
    data = load_and_prepare()
    for tf, df in data.items():
        print(f"\n📊 {tf.upper()} PREVIEW:")
        print(df.tail(3))
