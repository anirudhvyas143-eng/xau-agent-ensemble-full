# backtest_engine.py — simple next-close backtester for model features
import pandas as pd, joblib
from features_engineering import compute_indicators
from pathlib import Path

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"

def simple_backtest(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = compute_indicators(df)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
    df = df.dropna(subset=features + ["target"])
    split = int(len(df)*0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train[features])
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(scaler.transform(train[features]), train["target"])
    preds = model.predict(scaler.transform(test[features]))
    test = test.copy()
    test["pred"] = preds
    # simulate 1 unit per predicted buy (long-only for simplicity)
    test["strategy_returns"] = test["Close"].pct_change().shift(-1) * test["pred"]
    test["market_returns"] = test["Close"].pct_change().shift(-1)
    strategy_perf = (1 + test["strategy_returns"].fillna(0)).cumprod()
    market_perf = (1 + test["market_returns"].fillna(0)).cumprod()
    return {"strategy": float(strategy_perf.iloc[-1]), "market": float(market_perf.iloc[-1])}

if __name__ == "__main__":
    r = simple_backtest(DATA_DIR / "XAU_USD_Historical_Data_daily.csv")
    print("Backtest result (growth):", r)
