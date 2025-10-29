# main.py
# XAUUSD Ensemble Agent — Hourly + Daily signals with feature engineering, tuning, backtest, drift detection
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import joblib, json, threading, time, os, requests
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import warning
warnings.filterwarnings("ignore")

app = Flask(__name__)

# === Paths & config ===
ROOT = Path(".").resolve()
DAILY_FILE = ROOT / "features_full_daily.csv"
HOURLY_FILE = ROOT / "features_full_hourly.csv"
WEEKLY_FILE = ROOT / "features_full_weekly.csv"
MONTHLY_FILE = ROOT / "features_full_monthly.csv"
MODEL_HOUR = ROOT / "model_hour.pkl"
MODEL_DAY = ROOT / "model_day.pkl"
SCALER_HOUR = ROOT / "scaler_hour.pkl"
SCALER_DAY = ROOT / "scaler_day.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"
BACKTEST_DIR = ROOT / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)

# Run every hour
REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))

# Symbol: XAUUSD via Yahoo mapping (Gold futures or spot)
YF_SYMBOL = "GC=F"  # futures; if you prefer spot, change to "XAUUSD=X" (availability varies)

# === Utility: indicators ===
def compute_indicators(df):
    df = df.copy()
    # ensure index is datetime
    if "Date" in df.columns:
        df = df.rename(columns={"Date":"date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    # EMAs
    for span in (8, 21, 50, 200):
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    # ATR (True Range then EMA)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    # RSI(14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    df["rsi14"] = 100 - (100 / (1 + ma_up/ma_down))
    # Volatility / momentum
    df["ret1"] = df["Close"].pct_change()
    df["vol10"] = df["ret1"].rolling(10).std()
    df["mom5"] = df["Close"].pct_change(5)
    df = df.dropna()
    return df.reset_index()

# === Data fetching and resampling ===
def fetch_and_build_datasets():
    """Fetch hourly and daily OHLCV and compute features for hourly and daily models.
       Also build weekly and monthly from daily for context."""
    print("Fetching data from Yahoo Finance...")
    # Hourly: last 60 days of hourly bars (for speed)
    try:
        df_hour = yf.download(YF_SYMBOL, period="60d", interval="1h", progress=False)
            # 🔧 Normalize Yahoo Finance columns
    if not df_hour.empty:
        df_hour.columns = [c.capitalize() for c in df_hour.columns]
        df_hour.reset_index(inplace=True)
        print("✅ Hourly columns normalized:", df_hour.columns.tolist())
    except Exception as e:
        print("yfinance hourly fetch failed:", e)
        df_hour = pd.DataFrame()
    # Daily: longer history
    try:
        df_day = yf.download(YF_SYMBOL, period="10y", interval="1d", progress=False)
            # 🔧 Normalize daily data columns
    if not df_day.empty:
        df_day.columns = [c.capitalize() for c in df_day.columns]
        df_day.reset_index(inplace=True)
        print("✅ Daily columns normalized:", df_day.columns.tolist())
    except Exception as e:
        print("yfinance daily fetch failed:", e)
        df_day = pd.DataFrame()

    # Basic checks
    if df_hour.empty and df_day.empty:
        raise RuntimeError("Failed to fetch any data from yfinance.")

    # Compute features if available
    if not df_hour.empty:
        df_hour_proc = compute_indicators(df_hour.reset_index())
        df_hour_proc.to_csv(HOURLY_FILE, index=False)
        print("Saved hourly features:", HOURLY_FILE)

    if not df_day.empty:
        df_day_proc = compute_indicators(df_day.reset_index())
        df_day_proc.to_csv(DAILY_FILE, index=False)
        # weekly/monthly derived
        df_day_proc.index = pd.to_datetime(df_day_proc["date"])
        df_week = df_day_proc.resample("W").agg({
            "Open":"first","High":"max","Low":"min","Close":"last"
        }).dropna()
        df_week = compute_indicators(df_week.reset_index())
        df_week.to_csv(WEEKLY_FILE, index=False)
        df_month = df_day_proc.resample("M").agg({
            "Open":"first","High":"max","Low":"min","Close":"last"
        }).dropna()
        df_month = compute_indicators(df_month.reset_index())
        df_month.to_csv(MONTHLY_FILE, index=False)
        print("Saved daily/weekly/monthly features.")

    return

# === Feature engineering for model input ===
def build_feature_matrix_hourly():
    df_h = pd.read_csv(HOURLY_FILE, parse_dates=["date"])
    # Label: next-hour return positive?
    df_h["target"] = (df_h["Close"].shift(-1) > df_h["Close"]).astype(int)
    # Regime detection (unsupervised)
    regime_feats = ["vol10","mom5","rsi14"]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_h["regime"] = kmeans.fit_predict(df_h[regime_feats].fillna(0))
    # Feature list (hour model uses short-term indicators)
    feature_cols = ["ema8","ema21","ema50","atr14","rsi14","vol10","mom5","regime"]
    df_h = df_h.dropna(subset=feature_cols + ["target"])
    X = df_h[feature_cols]
    y = df_h["target"]
    return X, y, df_h

def build_feature_matrix_daily():
    df_d = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    df_w = pd.read_csv(WEEKLY_FILE, parse_dates=["date"]) if WEEKLY_FILE.exists() else None
    df_m = pd.read_csv(MONTHLY_FILE, parse_dates=["date"]) if MONTHLY_FILE.exists() else None

    # target: next-day return positive?
    df_d["target"] = (df_d["Close"].shift(-1) > df_d["Close"]).astype(int)

    # merge weekly/monthly latest values as context (by aligning last known to daily row)
    # simple approach: pad weekly/monthly values forward to daily index
    if df_w is not None:
        df_w_idx = df_w.set_index("date").reindex(df_d["date"], method="ffill").reset_index(drop=True)
        df_d["ema21_w"] = df_w_idx["ema21"]
        df_d["ema50_w"] = df_w_idx["ema50"]
        df_d["atr14_w"] = df_w_idx["atr14"]
    else:
        df_d["ema21_w"] = np.nan; df_d["ema50_w"]=np.nan; df_d["atr14_w"]=np.nan

    if df_m is not None:
        df_m_idx = df_m.set_index("date").reindex(df_d["date"], method="ffill").reset_index(drop=True)
        df_d["ema21_m"] = df_m_idx["ema21"]
        df_d["ema50_m"] = df_m_idx["ema50"]
        df_d["atr14_m"] = df_m_idx["atr14"]
    else:
        df_d["ema21_m"] = np.nan; df_d["ema50_m"]=np.nan; df_d["atr14_m"]=np.nan

    # Feature columns for daily model
    feature_cols = [
        "ema21","ema50","atr14","rsi14","vol10","mom5",
        "ema21_w","ema50_w","atr14_w","ema21_m","ema50_m","atr14_m"
    ]
    df_d = df_d.dropna(subset=feature_cols + ["target"])
    X = df_d[feature_cols]
    y = df_d["target"]
    return X, y, df_d

# === Model training with modest hyperparameter tuning ===
def train_model(X, y, model_path, scaler_path, tune=True, n_iter=20):
    """Train and save a pipeline: scaler + RandomForest with RandomizedSearch."""
    # split time-series style (no shuffle)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # baseline model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    if tune and len(X_train) > 50:
        param_dist = {
            "n_estimators": [100,200,300,400],
            "max_depth": [5,8,12,16,None],
            "min_samples_split": [2,5,8],
            "min_samples_leaf": [1,2,4]
        }
        rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=min(n_iter,20),
                                scoring="accuracy", cv=3, verbose=0, random_state=42)
        rs.fit(X_train_s, y_train)
        best = rs.best_estimator_
        model = best
    else:
        rf.fit(X_train_s, y_train)
        model = rf

    # final evaluate
    y_pred = model.predict(X_val_s)
    acc = accuracy_score(y_val, y_pred)
    print(f"Trained model accuracy on holdout: {acc:.3f}")

    # Save model and scaler as pipeline
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler, acc

# === Backtest simple strategy ===
def backtest(df, signal_col="signal", price_col="Close"):
    """Compute returns for a 1-step-ahead signal stored in df['signal'] (1 buy, 0 hold/sell)"""
    df = df.copy().reset_index(drop=True)
    # create position = previous signal (we act at next bar open/close)
    df["position"] = df[signal_col].shift(0)  # assume signal indicates direction at same bar
    df["return"] = df[price_col].pct_change().fillna(0)
    df["strategy_ret"] = df["position"] * df["return"]
    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
    df["cum_market"] = (1 + df["return"]).cumprod()
    # save plot
    plt.figure(figsize=(8,5))
    plt.plot(df["date"], df["cum_market"], label="Market")
    plt.plot(df["date"], df["cum_strategy"], label="Strategy")
    plt.legend()
    plt.title("Backtest: Strategy vs Market")
    out_png = BACKTEST_DIR / f"backtest_{int(time.time())}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    # simple metrics
    total_strat = df["cum_strategy"].iloc[-1] - 1
    total_mkt = df["cum_market"].iloc[-1] - 1
    return {"total_strategy_return": float(total_strat), "total_market_return": float(total_mkt), "plot": str(out_png)}

# === Drift detection (distribution shift) ===
def detect_drift(train_X, recent_X):
    """KS-test on each feature, return features with p < 0.01 as drifted."""
    drifted = {}
    for c in train_X.columns:
        try:
            stat, p = ks_2samp(train_X[c].dropna(), recent_X[c].dropna())
            drifted[c] = float(p)
        except Exception:
            drifted[c] = None
    return drifted

# === Full pipeline: train both models & produce signals ===
def build_train_and_signal():
    try:
        # fetch/process data
        fetch_and_build_datasets()

        # Hourly pipeline
        if HOURLY_FILE.exists():
            Xh, yh, dfh = build_feature_matrix_hourly()
            if len(Xh) > 50:
                model_h, scaler_h, acc_h = train_model(Xh, yh, MODEL_HOUR, SCALER_HOUR, tune=True, n_iter=10)
            else:
                model_h, scaler_h = train_model(Xh, yh, MODEL_HOUR, SCALER_HOUR, tune=False)
        else:
            Xh = yh = dfh = None
            model_h = scaler_h = None

        # Daily pipeline
        if DAILY_FILE.exists():
            Xd, yd, dfd = build_feature_matrix_daily()
            if len(Xd) > 50:
                model_d, scaler_d, acc_d = train_model(Xd, yd, MODEL_DAY, SCALER_DAY, tune=True, n_iter=10)
            else:
                model_d, scaler_d = train_model(Xd, yd, MODEL_DAY, SCALER_DAY, tune=False)
        else:
            Xd = yd = dfd = None
            model_d = scaler_d = None

        # Produce predictions using most recent rows
        results = {}
        # Hour prediction
        if model_h is not None and HOURLY_FILE.exists():
            last_h = dfh.iloc[-1:]
            X_last_h = last_h[["ema8","ema21","ema50","atr14","rsi14","vol10","mom5","regime"]]
            X_last_h_s = scaler_h.transform(X_last_h)
            prob_h = float(model_h.predict_proba(X_last_h_s)[0][1]) if hasattr(model_h,"predict_proba") else 0.0
            pred_h = int(model_h.predict(X_last_h_s)[0])
            results["hour_signal"] = "BUY" if pred_h==1 else "SELL"
            results["hour_confidence"] = round(prob_h*100,2)
        else:
            results["hour_signal"] = "N/A"; results["hour_confidence"] = 0.0

        # Day prediction
        if model_d is not None and DAILY_FILE.exists():
            last_d = dfd.iloc[-1:]
            feature_cols_d = [
                "ema21","ema50","atr14","rsi14","vol10","mom5",
                "ema21_w","ema50_w","atr14_w","ema21_m","ema50_m","atr14_m"
            ]
            X_last_d = last_d[feature_cols_d]
            X_last_d_s = scaler_d.transform(X_last_d)
            prob_d = float(model_d.predict_proba(X_last_d_s)[0][1]) if hasattr(model_d,"predict_proba") else 0.0
            pred_d = int(model_d.predict(X_last_d_s)[0])
            results["day_signal"] = "BUY" if pred_d==1 else "SELL"
            results["day_confidence"] = round(prob_d*100,2)
        else:
            results["day_signal"] = "N/A"; results["day_confidence"] = 0.0

        # Backtests
        if dfh is not None:
            # create a historical signal column using model predictions for backtest
            Xh_all = dfh[["ema8","ema21","ema50","atr14","rsi14","vol10","mom5","regime"]]
            Xh_s = scaler_h.transform(Xh_all)
            dfh["signal"] = model_h.predict(Xh_s)
            bt_hour = backtest(dfh, "signal", "Close")
            results["backtest_hour"] = bt_hour
        if dfd is not None:
            Xd_all = dfd[feature_cols_d]
            Xd_s = scaler_d.transform(Xd_all)
            dfd["signal"] = model_d.predict(Xd_s)
            bt_day = backtest(dfd, "signal", "Close")
            results["backtest_day"] = bt_day

        # Drift detection (compare recent 7 days/hours to training)
        drift = {}
        if Xh is not None and len(Xh) > 200:
            recent = Xh.tail(100)
            drift["hour_drift_pvalues"] = detect_drift(Xh.iloc[:int(len(Xh)*0.8)], recent)
        if Xd is not None and len(Xd) > 200:
            recentd = Xd.tail(60)
            drift["day_drift_pvalues"] = detect_drift(Xd.iloc[:int(len(Xd)*0.8)], recentd)

        # Save signals JSON & history
        out = {
            "timestamp": str(datetime.now(timezone.utc)),
            "hour_signal": results.get("hour_signal"),
            "hour_confidence": results.get("hour_confidence"),
            "day_signal": results.get("day_signal"),
            "day_confidence": results.get("day_confidence"),
            "drift": drift,
            "backtest_hour": results.get("backtest_hour"),
            "backtest_day": results.get("backtest_day")
        }
        with open(SIGNALS_FILE,"w") as f:
            json.dump(out, f, indent=2)

        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.load(open(HISTORY_FILE))
            except Exception:
                history = []
        history.append(out)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)

        print(f"[{out['timestamp']}] Hour:{out['hour_signal']}({out['hour_confidence']}%) Day:{out['day_signal']}({out['day_confidence']}%)")
        return out

    except Exception as e:
        print("Error in pipeline:", e)
        return {"error": str(e)}

# === Background scheduler with self-ping to keep Render awake ===
def background_loop():
    while True:
        try:
            build_train_and_signal()
            # self-ping (change to your render url if different)
            try:
                requests.get(os.getenv("SELF_PING_URL","https://xau-agent-ensemble-full.onrender.com/"), timeout=10)
            except Exception as e:
                print("Self-ping failed:", e)
        except Exception as e:
            print("Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)

# === Flask endpoints ===
@app.route("/")
def home():
    return jsonify({"status":"ok","time":str(datetime.now(timezone.utc))})

@app.route("/signal", methods=["GET"])
def signal_route():
    res = build_train_and_signal()
    return jsonify(res)

@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json()
    # Expect keys daily/hourly dicts with indicator names
    try:
        model_d = joblib.load(MODEL_DAY) if MODEL_DAY.exists() else None
        model_h = joblib.load(MODEL_HOUR) if MODEL_HOUR.exists() else None
        scaler_d = joblib.load(SCALER_DAY) if SCALER_DAY.exists() else None
        scaler_h = joblib.load(SCALER_HOUR) if SCALER_HOUR.exists() else None

        out = {}
        # hourly predict
        if "hourly" in payload and model_h is not None and scaler_h is not None:
            row = payload["hourly"]
            dfh = pd.DataFrame([row])[["ema8","ema21","ema50","atr14","rsi14","vol10","mom5","regime"]]
            ph = model_h.predict(scaler_h.transform(dfh))[0]
            php = float(model_h.predict_proba(scaler_h.transform(dfh))[0][1])
            out["hour_signal"] = "BUY" if ph==1 else "SELL"
            out["hour_confidence"] = round(php*100,2)
        # daily predict
        if "daily" in payload and model_d is not None and scaler_d is not None:
            row = payload["daily"]
            dfd = pd.DataFrame([row])[[
                "ema21","ema50","atr14","rsi14","vol10","mom5",
                "ema21_w","ema50_w","atr14_w","ema21_m","ema50_m","atr14_m"
            ]]
            pdp = float(model_d.predict_proba(scaler_d.transform(dfd))[0][1])
            pdp_label = int(model_d.predict(scaler_d.transform(dfd))[0])
            out["day_signal"] = "BUY" if pdp_label==1 else "SELL"
            out["day_confidence"] = round(pdp*100,2)
        return jsonify({"status":"ok","result":out})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400

@app.route("/history", methods=["GET"])
def history_route():
    if HISTORY_FILE.exists():
        return jsonify(json.load(open(HISTORY_FILE)))
    return jsonify([])

@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        current = json.load(open(SIGNALS_FILE))
    except Exception:
        current = {"hour_signal":"N/A","day_signal":"N/A","hour_confidence":0,"day_confidence":0}
    # minimal HTML dashboard
    return f"""
    <html><head><title>XAU Agent</title><meta http-equiv="refresh" content="300"></head>
    <body style="font-family:Arial;background:#0d1117;color:#fff;text-align:center;padding:40px;">
    <h1>XAU/USD Agent</h1>
    <div style="display:inline-block;background:#161b22;padding:20px;border-radius:12px;">
      <h2>Hourly: <span style="color:{'#0f0' if current.get('hour_signal')=='BUY' else '#f55'}">{current.get('hour_signal')}</span></h2>
      <p>Confidence: {current.get('hour_confidence')}%</p>
      <h2>Daily: <span style="color:{'#0f0' if current.get('day_signal')=='BUY' else '#f55'}">{current.get('day_signal')}</span></h2>
      <p>Confidence: {current.get('day_confidence')}%</p>
      <p><button onclick="fetch('/signal').then(()=>location.reload())">Regenerate</button></p>
    </div>
    </body></html>
    """

# === Start background thread and Flask app ===
if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
