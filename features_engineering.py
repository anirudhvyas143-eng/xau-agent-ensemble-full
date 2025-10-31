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

# =====================================================
# 🧩 Robust Column Detection
# =====================================================
def detect_columns(df):
    """Auto-detect Close, High, Low columns (case-insensitive) from any data source."""
    mapping = {
        "close": ["close", "Close", "Adj Close", "Price", "Last"],
        "high": ["high", "High"],
        "low": ["low", "Low"]
    }

    normalized = {}
    for key, variants in mapping.items():
        for v in variants:
            for col in df.columns:
                if v.lower() == col.lower():
                    normalized[key] = col
                    break
            if key in normalized:
                break

        # Fallback to close if high/low not available
        if key not in normalized:
            if key in ["high", "low"]:
                normalized[key] = "close"
            else:
                raise ValueError(f"❌ No '{key}' column found in dataframe. Columns: {df.columns.tolist()}")

    df["close"] = df[normalized["close"]]
    df["high"] = df[normalized["high"]]
    df["low"] = df[normalized["low"]]

    logger.info(f"✅ Using columns — Close: {normalized['close']}, High: {normalized['high']}, Low: {normalized['low']}")
    return df


# =====================================================
# 🧮 Technical Feature Computations
# =====================================================
def compute_technical_features(df):
    """Compute robust technical indicators for model training and live predictions."""
    df = df.copy()
    df = detect_columns(df)

    # === EMAs ===
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()

    # === RSI ===
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # === MACD + Signal Line ===
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

    # === ATR ===
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # === Volatility + Momentum ===
    df["volatility"] = df["close"].pct_change().rolling(10).std()
    df["momentum"] = df["close"] - df["close"].shift(10)

    df.dropna(inplace=True)
    logger.info(f"✅ Technical features computed. {len(df)} rows ready.")
    return df


# =====================================================
# 🧠 Multi-Timeframe Feature Fusion
# =====================================================
def merge_multi_timeframes(datasets):
    """Merge daily, weekly, and monthly (if any) features into one ensemble dataset."""
    df_daily = datasets.get("daily")
    df_weekly = datasets.get("weekly")
    df_monthly = datasets.get("monthly")

    if df_daily is None:
        logger.warning("⚠️ Missing daily data — cannot build ensemble.")
        return None

    if df_weekly is not None:
        df_weekly = df_weekly.resample("1D").ffill().add_suffix("_w")
    if df_monthly is not None:
        df_monthly = df_monthly.resample("1D").ffill().add_suffix("_m")

    combined = df_daily.copy()
    if df_weekly is not None:
        combined = combined.join(df_weekly, how="left")
    if df_monthly is not None:
        combined = combined.join(df_monthly, how="left")

    combined.dropna(inplace=True)
    logger.info(f"✅ Unified ensemble dataset ready: {combined.shape}")
    return combined


# =====================================================
# 🏗️ Main Feature Builder
# =====================================================
def build_features():
    """Load data (from CSV or live API), compute all technical indicators, and save outputs."""
    logger.info("🔄 Starting feature generation pipeline ...")
    datasets = load_and_prepare()
    os.makedirs("data", exist_ok=True)

    for tf, df in datasets.items():
        logger.info(f"🧮 Computing technical features for {tf} timeframe ({len(df)} rows)")
        try:
            df = compute_technical_features(df)
            out_path = OUTPUT_PATHS.get(tf, f"data/features_full_{tf}.csv")
            df.to_csv(out_path, index=True)
            logger.info(f"✅ Saved {tf} features → {out_path}")
        except Exception as e:
            logger.error(f"⚠️ Error computing {tf} features: {e}")

    ensemble_df = merge_multi_timeframes(datasets)
    if ensemble_df is not None:
        ensemble_df.to_csv(OUTPUT_PATHS["ensemble"], index=True)
        logger.info(f"🎯 Final ensemble features saved → {OUTPUT_PATHS['ensemble']}")
    else:
        logger.warning("⚠️ Ensemble dataset could not be created.")


if __name__ == "__main__":
    build_features()
    logger.info("✅ Feature generation completed successfully.")
