# main.py — XAUUSD Agent (Auto-growing hourly dataset + 25y daily fetch, indicators, training & signals)
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import joblib, json, threading, time, os, requests
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ---------------- CONFIG ----------------
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

REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))  # 1 hour default
YF_SYMBOL = os.getenv("YF_SYMBOL", "GC=F")  # Gold futures symbol
PORT = int(os.getenv("PORT", 10000))
SELF_PING_URL = os.getenv("SELF_PING_URL", None)


# ---------------- UTILITIES ----------------
def normalize_ohlcv_df(df):
    """Normalize yfinance DataFrame to always have Open/High/Low/Close/Volume columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ["open", "o"]:
            rename_map[col] = "Open"
        elif cl in ["high", "h"]:
            rename_map[col] = "High"
        elif cl in ["low", "l"]:
            rename_map[col] = "Low"
        elif cl in ["close", "adjclose", "adj close", "adj_close", "c", "last", "price"]:
            rename_map[col] = "Close"
        elif cl in ["volume", "v"]:
            rename_map[col] = "Volume"

    df.rename(columns=rename_map, inplace=True)

    # If Close missing but Adj Close exists, fallback
    if "Close" not in df.columns:
        for alt in ["Adj Close", "Adj_Close", "adjclose"]:
            if alt in df.columns:
                df["Close"] = df[alt]
                break

    # ensure Date column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.columns[0]: "Date"})
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with missing close data
    df = df.dropna(subset=["Close"])
    return df


def compute_indicators(df):
    """Compute indicators safely. Returns DataFrame or empty if data missing."""
    try:
        df = normalize_ohlcv_df(df)
        if "Close" not in df.columns:
            print("⚠️ No Close column found, skipping indicator computation.")
            return pd.DataFrame()

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")

        # EMA
        for span in (8, 21, 50, 200):
            df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

        # ATR
        if {"High", "Low", "Close"}.issubset(df.columns):
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14).mean()
        else:
            df["atr14"] = df["Close"].pct_change().rolling(14).std() * df["Close"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # Momentum & Volatility
        df["mom5"] = df["Close"].pct_change(5)
        df["vol10"] = df["Close"].pct_change().rolling(10).std()
        df = df.dropna().reset_index()
        return df
    except Exception as e:
        print("⚠️ Indicator computation error:", e)
        return pd.DataFrame()


def fetch_yf(symbol, period, interval):
    """Download data with yfinance."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return normalize_ohlcv_df(df)
    except Exception as e:
        print(f"⚠️ yfinance error ({symbol}, {period}, {interval}):", e)
        return pd.DataFrame()


# ---------------- FETCH + BUILD ----------------
def fetch_and_build_datasets():
    print("📥 Fetching data from Yahoo Finance...")
    df_day = pd.DataFrame()
    for p in ["25y", "15y", "10y", "5y"]:
        df_day = fetch_yf(YF_SYMBOL, period=p, interval="1d")
        if not df_day.empty:
            print(f"✅ Fetched daily ({p}) rows={len(df_day)}")
            break

    df_hour = pd.DataFrame()
    for p in ["5y", "2y", "1y", "60d"]:
        df_hour = fetch_yf(YF_SYMBOL, period=p, interval="1h")
        if not df_hour.empty:
            print(f"✅ Fetched hourly ({p}) rows={len(df_hour)}")
            break

    if df_day.empty and df_hour.empty:
        raise RuntimeError("❌ Failed to fetch any valid data from Yahoo Finance.")

    # Daily pipeline
    if not df_day.empty:
        try:
            df_day_proc = compute_indicators(df_day)
            if not df_day_proc.empty:
                df_day_proc.to_csv(DAILY_FILE, index=False)
                print("💾 Saved daily features:", DAILY_FILE)
                # weekly + monthly
                df_day_proc.index = pd.to_datetime(df_day_proc["Date"])
                df_week = df_day_proc.resample("W").agg({
                    "Open": "first", "High": "max", "Low": "min", "Close": "last"
                }).dropna()
                if not df_week.empty:
                    compute_indicators(df_week.reset_index()).to_csv(WEEKLY_FILE, index=False)
                    print("💾 Saved weekly features:", WEEKLY_FILE)
                df_month = df_day_proc.resample("M").agg({
                    "Open": "first", "High": "max", "Low": "min", "Close": "last"
                }).dropna()
                if not df_month.empty:
                    compute_indicators(df_month.reset_index()).to_csv(MONTHLY_FILE, index=False)
                    print("💾 Saved monthly features:", MONTHLY_FILE)
        except Exception as e:
            print("⚠️ Error processing daily data:", e)

    # Hourly pipeline
    if not df_hour.empty:
        try:
            df_hour_proc = compute_indicators(df_hour)
            if df_hour_proc.empty:
                print("⚠️ Hourly indicator computation returned empty dataset.")
                return
            if HOURLY_FILE.exists():
                existing = pd.read_csv(HOURLY_FILE, parse_dates=["Date"])
                combined = pd.concat([existing, df_hour_proc]).drop_duplicates(subset=["Date"], keep="last")
                combined = combined.sort_values("Date").reset_index(drop=True)
                combined.to_csv(HOURLY_FILE, index=False)
                print(f"📈 Appended hourly data → {len(combined)} rows total.")
            else:
                df_hour_proc.to_csv(HOURLY_FILE, index=False)
                print("💾 Saved hourly features:", HOURLY_FILE)
        except Exception as e:
            print("⚠️ Error processing hourly data:", e)


# ---------------- MODEL TRAINING ----------------
def train_simple_model(X, y, model_path):
    if len(X) < 30:
        return None
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)
    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_path).replace(".pkl", "_scaler.pkl"))
    print(f"🤖 Trained model saved at {model_path} (val acc={acc:.3f})")
    return model


# ---------------- PIPELINE ----------------
def build_train_and_signal():
    try:
        fetch_and_build_datasets()
    except Exception as e:
        print("Fetch error:", e)

    results = {}

    # HOURLY
    if HOURLY_FILE.exists():
        try:
            dfh = pd.read_csv(HOURLY_FILE, parse_dates=["Date"])
            dfh["target"] = (dfh["Close"].shift(-1) > dfh["Close"]).astype(int)
            features_h = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
            dfh = dfh.dropna(subset=features_h + ["target"])
            if len(dfh) >= 40:
                Xh, yh = dfh[features_h], dfh["target"]
                model_h = train_simple_model(Xh, yh, MODEL_HOUR)
                scaler_h = joblib.load(str(MODEL_HOUR).replace(".pkl", "_scaler.pkl"))
                last = Xh.iloc[[-1]]
                prob = float(model_h.predict_proba(scaler_h.transform(last))[0][1])
                pred = int(model_h.predict(scaler_h.transform(last))[0])
                results["hour_signal"] = "BUY" if pred == 1 else "SELL"
                results["hour_confidence"] = round(prob * 100, 2)
            else:
                results["hour_signal"], results["hour_confidence"] = "N/A", 0.0
        except Exception as e:
            print("⚠️ Hourly model error:", e)
            results["hour_signal"], results["hour_confidence"] = "N/A", 0.0

    # DAILY
    if DAILY_FILE.exists():
        try:
            dfd = pd.read_csv(DAILY_FILE, parse_dates=["Date"])
            dfd["target"] = (dfd["Close"].shift(-1) > dfd["Close"]).astype(int)
            features_d = ["ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
            dfd = dfd.dropna(subset=features_d + ["target"])
            if len(dfd) >= 50:
                Xd, yd = dfd[features_d], dfd["target"]
                model_d = train_simple_model(Xd, yd, MODEL_DAY)
                scaler_d = joblib.load(str(MODEL_DAY).replace(".pkl", "_scaler.pkl"))
                last = Xd.iloc[[-1]]
                prob = float(model_d.predict_proba(scaler_d.transform(last))[0][1])
                pred = int(model_d.predict(scaler_d.transform(last))[0])
                results["day_signal"] = "BUY" if pred == 1 else "SELL"
                results["day_confidence"] = round(prob * 100, 2)
            else:
                results["day_signal"], results["day_confidence"] = "N/A", 0.0
        except Exception as e:
            print("⚠️ Daily model error:", e)
            results["day_signal"], results["day_confidence"] = "N/A", 0.0

    # save
    out = {
        "timestamp": str(datetime.now(timezone.utc)),
        **results
    }
    json.dump(out, open(SIGNALS_FILE, "w"), indent=2)
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
        except Exception:
            history = []
    history.append(out)
    json.dump(history, open(HISTORY_FILE, "w"), indent=2)

    print(f"[{out['timestamp']}] Hour:{out.get('hour_signal')}({out.get('hour_confidence')}%) "
          f"Day:{out.get('day_signal')}({out.get('day_confidence')}%)")
    return out


# ---------------- BACKGROUND LOOP ----------------
def background_loop():
    while True:
        try:
            build_train_and_signal()
            if SELF_PING_URL:
                try:
                    requests.get(SELF_PING_URL, timeout=8)
                except Exception:
                    pass
        except Exception as e:
            print("Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)


# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})


@app.route("/signal")
def signal_route():
    return jsonify(build_train_and_signal())


@app.route("/history")
def history_route():
    if HISTORY_FILE.exists():
        try:
            return jsonify(json.load(open(HISTORY_FILE)))
        except Exception:
            return jsonify([])
    return jsonify([])


@app.route("/dashboard")
def dashboard():
    try:
        current = json.load(open(SIGNALS_FILE))
    except Exception:
        current = {"hour_signal": "N/A", "day_signal": "N/A", "hour_confidence": 0, "day_confidence": 0}

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


# ---------------- START ----------------
if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    print(f"🚀 Starting Flask on port {PORT} | Refresh interval {REFRESH_INTERVAL_SECS}s | Symbol {YF_SYMBOL}")
    app.run(host="0.0.0.0", port=PORT)
