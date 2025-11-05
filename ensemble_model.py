import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------
# Utility: compute indicators
# -----------------------------------------------------
def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("Missing Close column in dataset")

    df["ema8"] = df["Close"].ewm(span=8).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["atr14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["rsi14"] = compute_rsi(df["Close"], 14)
    df["mom5"] = df["Close"].diff(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -----------------------------------------------------
# Ensemble signal generator
# -----------------------------------------------------
def generate_ensemble_signal(df: pd.DataFrame, timeframe="daily"):
    df = compute_indicators(df)
    features = ["ema8", "ema21", "ema50", "atr14", "rsi14", "mom5", "vol10"]
    df = df.dropna(subset=features)
    if len(df) < 200:
        return {"signal": "N/A", "confidence": 0}

    # Target variable: 1 if next close > current close
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[features]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train small ensemble
    rf = RandomForestClassifier(n_estimators=80, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lgb = LGBMClassifier(n_estimators=80, random_state=42, verbose=-1)

    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    lgb.fit(X_scaled, y)

    last = X_scaled[-1].reshape(1, -1)
    p1 = rf.predict_proba(last)[0][1]
    p2 = gb.predict_proba(last)[0][1]
    p3 = lgb.predict_proba(last)[0][1]

    avg_prob = np.mean([p1, p2, p3])
    signal = "BUY" if avg_prob > 0.55 else "SELL" if avg_prob < 0.45 else "HOLD"

    print(f"✅ [{timeframe.upper()}] ensemble signal = {signal} ({avg_prob:.2f})")

    return {"signal": signal, "confidence": round(abs(avg_prob - 0.5) * 200, 2)}
