# main.py
# XAUUSD Agent — Auto-growing hourly dataset + 25y daily fetch, indicators, train & signal
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import joblib, json, threading, time, os, requests
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# -------- CONFIG --------
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
HOURLY_FILE = ROOT / "features_full_hourly.csv"
DAILY_FILE = ROOT / "features_full_daily.csv"
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

# Use environment variables where available
REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))  # default 1 hour
YF_SYMBOL = os.getenv("YF_SYMBOL", "GC=F")  # gold futures default; try "XAUUSD=X" if preferred
PORT = int(os.getenv("PORT", 10000))
SELF_PING_URL = os.getenv("SELF_PING_URL", None)

# -------- Utility functions --------
def normalize_ohlcv_df(df):
    """
    Normalize column names to Title-case OHLC: 'Open','High','Low','Close','Volume','Date'
    Accepts yfinance DataFrames or others.
    """
    if df is None or df.empty:
        return df
    # If yfinance returns a MultiIndex columns for adj close etc, flatten
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    cols = {c: c.strip() for c in df.columns}
    # common candidates for date column
    if "Datetime" in df.columns or "Date" in df.columns:
        pass  # keep index or column as is
    # rename to Title-case OHLC if lowercase present
    mapping = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "o"):
            mapping[c] = "Open"
        elif cl in ("high", "h"):
            mapping[c] = "High"
        elif cl in ("low", "l"):
            mapping[c] = "Low"
        elif cl in ("close", "c", "adjclose", "adj_close", "adj close", "close_adj"):
            mapping[c] = "Close"
        elif cl in ("volume", "v"):
            mapping[c] = "Volume"
        elif "date" in cl or "datetime" in cl or "time" in cl:
            mapping[c] = "Date"
    if mapping:
        df = df.rename(columns=mapping)
    # if index is datetime, move to column 'Date'
    if isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
        df = df.reset_index().rename(columns={df.reset_index().columns[0]: "Date"})
    # ensure Date column is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def compute_indicators(df):
    """Compute EMAs, ATR, RSI, volatility, momentum. Expects columns Open/High/Low/Close (title-case)."""
    df = df.copy()
    # ensure Date exists
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "date"})
    else:
        # try index as datetime
        try:
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
        except Exception:
            pass

    # normalize close column name case-insensitive
    for col in list(df.columns):
        if col.lower() == "close" and col != "Close":
            df = df.rename(columns={col: "Close"})
        if col.lower() == "open" and col != "Open":
            df = df.rename(columns={col: "Open"})
        if col.lower() == "high" and col != "High":
            df = df.rename(columns={col: "High"})
        if col.lower() == "low" and col != "Low":
            df = df.rename(columns={col: "Low"})

    # Require Close; if not present, return empty
    if "Close" not in df.columns:
        raise ValueError("No Close column available for indicator computation")

    # Make sure date is datetime and set index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    # calculate EMAs
    for span in (8, 21, 50, 200):
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # ATR (True Range -> EMA)
    if {"High", "Low", "Close"}.issubset(df.columns):
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    else:
        # fallback to using close-based volatility as proxy
        df["atr14"] = df["Close"].pct_change().rolling(14).std() * df["Close"]

    # RSI(14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    df["rsi14"] = 100 - (100 / (1 + ma_up / (ma_down.replace(0, np.nan))))

    # Volatility & momentum
    df["ret1"] = df["Close"].pct_change()
    df["vol10"] = df["ret1"].rolling(10).std()
    df["mom5"] = df["Close"].pct_change(5)

    df = df.dropna()
    df = df.reset_index().rename(columns={"date": "date"})
    return df

# -------- Data fetch & auto-append logic --------
def fetch_yf(symbol, period, interval):
    """Attempt to download with yfinance and normalize columns."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        df = normalize_ohlcv_df(df)
        return df
    except Exception as e:
        print(f"yfinance download error ({symbol}, {period}, {interval}):", e)
        return pd.DataFrame()

def fetch_and_build_datasets():
    """
    Fetch daily (attempt 25y) and hourly (attempt 5y; fallback to smaller if Yahoo denies),
    compute indicators and save features. Hourly file auto-grows by appending new unique rows.
    """
    print("Fetching data from Yahoo Finance...")
    # DAILY: try 25y (may be limited), fallback to 10y then 5y
    daily_periods = ["25y", "15y", "10y", "5y"]
    df_day = pd.DataFrame()
    for p in daily_periods:
        df_day = fetch_yf(YF_SYMBOL, period=p, interval="1d")
        if not df_day.empty:
            print(f"Fetched daily with period={p}, rows={len(df_day)}")
            break

    # HOURLY: try 5y, fallback to 2y, 1y, 60d
    hourly_periods = ["5y", "2y", "1y", "60d"]
    df_hour = pd.DataFrame()
    for p in hourly_periods:
        df_hour = fetch_yf(YF_SYMBOL, period=p, interval="1h")
        if not df_hour.empty:
            print(f"Fetched hourly with period={p}, rows={len(df_hour)}")
            break

    if df_day.empty and df_hour.empty:
        raise RuntimeError("Failed to fetch any data from yfinance.")

    # Process daily
    if not df_day.empty:
        try:
            df_day_proc = compute_indicators(df_day.reset_index())
            df_day_proc.to_csv(DAILY_FILE, index=False)
            print("Saved daily features:", DAILY_FILE)
            # derive weekly/monthly from daily
            df_day_proc.index = pd.to_datetime(df_day_proc["date"])
            df_week_raw = df_day_proc.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            }).dropna()
            if not df_week_raw.empty:
                df_week = compute_indicators(df_week_raw.reset_index())
                df_week.to_csv(WEEKLY_FILE, index=False)
                print("Saved weekly features:", WEEKLY_FILE)
            df_month_raw = df_day_proc.resample("M").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            }).dropna()
            if not df_month_raw.empty:
                df_month = compute_indicators(df_month_raw.reset_index())
                df_month.to_csv(MONTHLY_FILE, index=False)
                print("Saved monthly features:", MONTHLY_FILE)
        except Exception as e:
            print("Error processing daily data:", e)

    # Process hourly and auto-append
    if not df_hour.empty:
        try:
            df_hour_proc = compute_indicators(df_hour.reset_index())
            # if hourly file already exists, append only new rows
            if HOURLY_FILE.exists():
                try:
                    existing = pd.read_csv(HOURLY_FILE, parse_dates=["date"])
                    # combine and dedupe by date
                    combined = pd.concat([existing, df_hour_proc], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                    combined.to_csv(HOURLY_FILE, index=False)
                    print(f"Appended hourly rows. Hourly now: {len(combined)} rows")
                except Exception as e:
                    print("Error appending hourly file, overwriting with fresh:", e)
                    df_hour_proc.to_csv(HOURLY_FILE, index=False)
                    print("Saved hourly features:", HOURLY_FILE)
            else:
                df_hour_proc.to_csv(HOURLY_FILE, index=False)
                print("Saved hourly features:", HOURLY_FILE)
        except Exception as e:
            print("Error processing hourly data:", e)

# -------- Simple train & predict helpers --------
def train_simple_model(X, y, model_path):
    if len(X) < 20:
        # too small to train reliably
        return None
    # time-based split
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)
    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_path).replace(".pkl", "_scaler.pkl"))
    print(f"Trained model saved at {model_path} (val acc {acc:.3f})")
    return model

def build_train_and_signal():
    """
    Top-level orchestration. Fetch/process data, train hour/daily models (fallback quick training),
    predict latest hour & day signal, save signals + history.
    """
    try:
        fetch_and_build_datasets()
    except Exception as e:
        print("Fetch error:", e)

    results = {}
    # HOURLY model pipeline
    if HOURLY_FILE.exists():
        try:
            dfh = pd.read_csv(HOURLY_FILE, parse_dates=["date"])
            # label: next-hour up?
            dfh["target"] = (dfh["Close"].shift(-1) > dfh["Close"]).astype(int)
            features_h = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
            dfh = dfh.dropna(subset=features_h + ["target"])
            if len(dfh) >= 30:
                Xh = dfh[features_h]
                yh = dfh["target"]
                model_h = train_simple_model(Xh, yh, MODEL_HOUR) or joblib.load(MODEL_HOUR) if MODEL_HOUR.exists() else None
                scaler_h = joblib.load(str(MODEL_HOUR).replace(".pkl", "_scaler.pkl")) if Path(str(MODEL_HOUR).replace(".pkl", "_scaler.pkl")).exists() else None
                if model_h is None and MODEL_HOUR.exists():
                    model_h = joblib.load(MODEL_HOUR)
                    scaler_h = joblib.load(str(MODEL_HOUR).replace(".pkl", "_scaler.pkl")) if Path(str(MODEL_HOUR).replace(".pkl", "_scaler.pkl")).exists() else None
                if model_h is not None and scaler_h is not None:
                    last_h = dfh.iloc[-1:][features_h]
                    Xs = scaler_h.transform(last_h)
                    prob = float(model_h.predict_proba(Xs)[0][1]) if hasattr(model_h, "predict_proba") else 0.0
                    pred = int(model_h.predict(Xs)[0])
                    results["hour_signal"] = "BUY" if pred == 1 else "SELL"
                    results["hour_confidence"] = round(prob * 100, 2)
                else:
                    results["hour_signal"] = "N/A"; results["hour_confidence"] = 0.0
            else:
                results["hour_signal"] = "N/A"; results["hour_confidence"] = 0.0
        except Exception as e:
            print("Hourly pipeline error:", e)
            results["hour_signal"] = "N/A"; results["hour_confidence"] = 0.0
    else:
        results["hour_signal"] = "N/A"; results["hour_confidence"] = 0.0

    # DAILY model pipeline
    if DAILY_FILE.exists():
        try:
            dfd = pd.read_csv(DAILY_FILE, parse_dates=["date"])
            # label next-day up?
            dfd["target"] = (dfd["Close"].shift(-1) > dfd["Close"]).astype(int)
            # try to pull weekly/monthly context if present
            if WEEKLY_FILE.exists():
                dfw = pd.read_csv(WEEKLY_FILE, parse_dates=["date"]).set_index("date")
                # forward-fill weekly values onto daily rows
                dfw_ff = dfw.reindex(dfd["date"], method="ffill").reset_index(drop=True)
                dfd["ema21_w"] = dfw_ff["ema21"]
                dfd["ema50_w"] = dfw_ff["ema50"]
                dfd["atr14_w"] = dfw_ff["atr14"]
            else:
                dfd["ema21_w"] = np.nan; dfd["ema50_w"] = np.nan; dfd["atr14_w"] = np.nan
            if MONTHLY_FILE.exists():
                dfm = pd.read_csv(MONTHLY_FILE, parse_dates=["date"]).set_index("date")
                dfm_ff = dfm.reindex(dfd["date"], method="ffill").reset_index(drop=True)
                dfd["ema21_m"] = dfm_ff["ema21"]
                dfd["ema50_m"] = dfm_ff["ema50"]
                dfd["atr14_m"] = dfm_ff["atr14"]
            else:
                dfd["ema21_m"] = np.nan; dfd["ema50_m"] = np.nan; dfd["atr14_m"] = np.nan

            feature_cols_d = [
                "ema21","ema50","atr14","rsi14","vol10","mom5",
                "ema21_w","ema50_w","atr14_w","ema21_m","ema50_m","atr14_m"
            ]
            dfd = dfd.dropna(subset=feature_cols_d + ["target"])
            if len(dfd) >= 40:
                Xd = dfd[feature_cols_d]
                yd = dfd["target"]
                model_d = train_simple_model(Xd, yd, MODEL_DAY) or joblib.load(MODEL_DAY) if MODEL_DAY.exists() else None
                scaler_d = joblib.load(str(MODEL_DAY).replace(".pkl", "_scaler.pkl")) if Path(str(MODEL_DAY).replace(".pkl", "_scaler.pkl")).exists() else None
                if model_d is None and MODEL_DAY.exists():
                    model_d = joblib.load(MODEL_DAY)
                    scaler_d = joblib.load(str(MODEL_DAY).replace(".pkl", "_scaler.pkl")) if Path(str(MODEL_DAY).replace(".pkl", "_scaler.pkl")).exists() else None
                if model_d is not None and scaler_d is not None:
                    last_d = dfd.iloc[-1:][feature_cols_d]
                    Xd_s = scaler_d.transform(last_d)
                    probd = float(model_d.predict_proba(Xd_s)[0][1]) if hasattr(model_d, "predict_proba") else 0.0
                    predd = int(model_d.predict(Xd_s)[0])
                    results["day_signal"] = "BUY" if predd == 1 else "SELL"
                    results["day_confidence"] = round(probd * 100, 2)
                else:
                    results["day_signal"] = "N/A"; results["day_confidence"] = 0.0
            else:
                results["day_signal"] = "N/A"; results["day_confidence"] = 0.0
        except Exception as e:
            print("Daily pipeline error:", e)
            results["day_signal"] = "N/A"; results["day_confidence"] = 0.0
    else:
        results["day_signal"] = "N/A"; results["day_confidence"] = 0.0

    # Save and return
    out = {
        "timestamp": str(datetime.now(timezone.utc)),
        "hour_signal": results.get("hour_signal"),
        "hour_confidence": results.get("hour_confidence"),
        "day_signal": results.get("day_signal"),
        "day_confidence": results.get("day_confidence"),
    }
    try:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(out, f, indent=2)
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.load(open(HISTORY_FILE))
            except Exception:
                history = []
        history.append(out)
        json.dump(history, open(HISTORY_FILE, "w"), indent=2)
    except Exception as e:
        print("Error saving signals/history:", e)

    print(f"[{out['timestamp']}] Hour:{out['hour_signal']}({out['hour_confidence']}%) Day:{out['day_signal']}({out['day_confidence']}%)")
    return out

# -------- Background runner --------
def background_loop():
    while True:
        try:
            build_train_and_signal()
            # optional self-ping to keep service awake
            if SELF_PING_URL:
                try:
                    requests.get(SELF_PING_URL, timeout=8)
                except Exception:
                    pass
        except Exception as e:
            print("Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)

# -------- Flask endpoints --------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})

@app.route("/signal", methods=["GET"])
def signal_route():
    res = build_train_and_signal()
    return jsonify(res)

@app.route("/history", methods=["GET"])
def history_route():
    if HISTORY_FILE.exists():
        try:
            return jsonify(json.load(open(HISTORY_FILE)))
        except Exception:
            return jsonify([])
    return jsonify([])

@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        current = json.load(open(SIGNALS_FILE))
    except Exception:
        current = {"hour_signal":"N/A","day_signal":"N/A","hour_confidence":0,"day_confidence":0}
    # simple dashboard
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

# -------- Start --------
if __name__ == "__main__":
    # start background thread
    threading.Thread(target=background_loop, daemon=True).start()
    print(f"Starting Flask on port {PORT} — refresh interval {REFRESH_INTERVAL_SECS}s — symbol {YF_SYMBOL}")
    app.run(host="0.0.0.0", port=PORT)
