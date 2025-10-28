import pandas as pd
import numpy as np
import os

# Paths to your datasets
DATA_DIR = "data"
FILES = {
    "daily": "XAU_USD_Historical_Data_daily.csv",
    "weekly": "XAU_USD_Historical_Data_weekly.csv",
    "monthly": "XAU_USD_Historical_Data_monthly.csv"
}

def load_and_prepare():
    datasets = {}
    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        print(f"Loading {tf} data from {path}...")
        df = pd.read_csv(path)

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Rename key columns if needed
        if "price" in df.columns:
            df.rename(columns={"price": "close"}, inplace=True)
        if "close" not in df.columns:
            df.rename(columns={"last": "close"}, inplace=True)
        
        # Convert date/time
        for col in ["date", "datetime", "time"]:
            if col in df.columns:
                df["date"] = pd.to_datetime(df[col])
                break

        df = df.sort_values("date")
        df = df.set_index("date")

        # Add simple features
        df["return"] = df["close"].pct_change()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["rsi_14"] = compute_rsi(df["close"], 14)
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        df["volatility"] = df["close"].pct_change().rolling(10).std()

        datasets[tf] = df.dropna()

    return datasets


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


if __name__ == "__main__":
    data = load_and_prepare()
    for tf, df in data.items():
        print(f"{tf} data: {len(df)} rows, columns: {df.columns.tolist()[:8]}")
        print(df.tail(3))
