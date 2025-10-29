import pandas as pd
import numpy as np
import os
import ta  # technical analysis library

# === CONFIG ===
DATA_DIR = "data"
FILES = {
    "daily": "XAU_USD_Historical_Data_daily.csv",
    "weekly": "XAU_USD_Historical_Data_weekly.csv",
    "monthly": "XAU_USD_Historical_Data_monthly.csv"
}


def load_and_prepare():
    """Load, clean, and feature-engineer multi-timeframe datasets."""
    datasets = {}

    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
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
