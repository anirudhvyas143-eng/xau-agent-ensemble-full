import pandas as pd
import numpy as np
import os
from load_data import load_and_prepare
from loguru import logger

# === CONFIG ===
OUTPUT_PATHS = {
    "hourly": "data/features_full_hourly.csv",
    "daily": "data/features_full_daily.csv",
    "weekly": "data/features_full_weekly.csv",
    "ensemble": "data/features_ensemble.csv"
}

def detect_close_column(df):
    """Auto-detect and normalize close column name from Yahoo or any source."""
    possible_close_cols = ["Close", "Adj Close", "close", "Price", "Last"]
    for c in possible_close_cols:
        if c in df.columns:
            df["close"] = df[c]
            break
    else:
        raise ValueError(f"❌ None of {possible_close_cols} found in columns: {df.columns.tolist()}")
    return df


def compute_technical_features(df):
    """Compute technical indicators."""
    df = df.copy()
    df = detect_close_column(df)

    if "High" in df.columns and "Low" in df.columns:
        df.rename(columns={"High": "high", "Low": "low"}, inplace=True)
    elif not all(col in df.columns for col in ["high", "low"]):
        raise ValueError("❌ High/Low columns missing for ATR computation")

    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["rsi_14"] = compute_rsi(df["close"])
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["atr_14"] = compute_atr(df)
    df["momentum"] = df["close"] - df["close"].shift(10)
    df["volatility"] = df["close"].pct_change().rolling(10).std()
    df.dropna(inplace=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def merge_multi_timeframes(datasets):
    """Merge daily, weekly, and monthly features."""
    df_daily = datasets.get("daily")
    df_weekly = datasets.get("weekly")
    df_monthly = datasets.get("monthly")

    if df_daily is None:
        logger.warning("⚠️ Missing daily data.")
        return None

    # Resample weekly & monthly to daily date index for alignment
    if df_weekly is not None:
        df_weekly = df_weekly.resample("1D").ffill()
        df_weekly = df_weekly.add_suffix("_w")
    if df_monthly is not None:
        df_monthly = df_monthly.resample("1D").ffill()
        df_monthly = df_monthly.add_suffix("_m")

    combined = df_daily.copy()
    if df_weekly is not None:
        combined = combined.join(df_weekly, how="left")
    if df_monthly is not None:
        combined = combined.join(df_monthly, how="left")

    combined.dropna(inplace=True)
    logger.info(f"✅ Unified ensemble features: {combined.shape}")
    return combined


def build_features():
    """Generate and save all technical + ensemble features."""
    logger.info("🔄 Loading base data and generating features...")
    datasets = load_and_prepare()

    os.makedirs("data", exist_ok=True)

    for tf, df in datasets.items():
        logger.info(f"🧮 Computing technical features for {tf} ({len(df)} rows)")
        df = compute_technical_features(df)
        out_path = OUTPUT_PATHS.get(tf, f"data/features_full_{tf}.csv")
        df.to_csv(out_path)
        logger.info(f"✅ Saved {tf} features → {out_path}")

    ensemble_df = merge_multi_timeframes(datasets)
    if ensemble_df is not None:
        ensemble_df.to_csv(OUTPUT_PATHS["ensemble"])
        logger.info(f"🎯 Saved final ensemble features → {OUTPUT_PATHS['ensemble']}")


if __name__ == "__main__":
    build_features()
    logger.info("✅ Feature generation complete.")
