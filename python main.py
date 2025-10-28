#!/usr/bin/env python3
"""
main.py
- Keeps a small Flask app alive for UptimeRobot pings.
- Periodically (default every 4 hours) runs feature creation, optional quick retrain, and inference.
- Writes signals.json with entry/exit levels and basic metadata.

Usage (Render):
    python main.py
Environment:
    INFER_INTERVAL_SECS - seconds between periodic inferences (default 4 hours)
    DATA_PATH - path to primary CSV in data/ (optional)
"""
import os, time, threading, json, logging
from datetime import datetime
from pathlib import Path

# minimal deps
try:
    from flask import Flask, jsonify
except Exception as e:
    raise RuntimeError("Flask is required. Add flask to requirements.txt") from e

try:
    import pandas as pd, numpy as np, joblib
except Exception as e:
    raise RuntimeError("pandas/numpy/joblib required. Add to requirements.txt") from e

from sklearn.ensemble import RandomForestClassifier

# Config
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
FEATURES_CSV = ROOT / "features_full_daily.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "rf_model.joblib"
SIGNALS_FILE = ROOT / "signals.json"
INFER_INTERVAL_SECS = int(os.getenv("INFER_INTERVAL_SECS", 4 * 3600))  # default 4 hours
DATA_PRIMARY = os.getenv("DATA_PATH", str(next(DATA_DIR.glob("*.csv")).name if DATA_DIR.exists() and any(DATA_DIR.glob("*.csv")) else ""))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("xau_agent_main")

# --- Utility: feature engineering (same logic as other scripts) ---
def safe_read(path):
    df = pd.read_csv(path, engine="python")
    # normalize timestamp column
    for c in df.columns:
        if c.lower() in ("timestamp", "date", "time", "datetime"):
            df = df.rename(columns={c: "timestamp"})
            break
    if "timestamp" not in df.columns:
        # try first column
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
    # normalize close column
    if "Close" not in df.columns and "close" in df.columns:
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if "Close" not in df.columns:
        for c in df.columns:
            if c.lower() in ("close", "price", "last"):
                df = df.rename(columns={c: "Close"})
                break
    # numeric conversions
    for col in ("Open","High","Low","Close","Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("%",""), errors="coerce")
    df = df.dropna(subset=["timestamp","Close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def compute_features_from_primary(primary_csv_path):
    """Loads primary CSV and computes standard features, saves to FEATURES_CSV."""
    log.info("Computing features from %s", primary_csv_path)
    df = safe_read(primary_csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["Close"].pct_change()
    df["ema8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean(); ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    df["rsi14"] = 100 - (100/(1 + ma_up/ma_down))
    # TR / ATR
    if "High" in df.columns and "Low" in df.columns:
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    else:
        df["atr14"] = df["ret"].rolling(14).std() * df["Close"]
    df["vol10"] = df["ret"].rolling(10).std()
    df["mom5"] = df["Close"].pct_change(5)
    df["ema_diff"] = df["ema21"] - df["ema200"]
    df = df.dropna().reset_index(drop=True)
    df.to_csv(FEATURES_CSV, index=False)
    log.info("Saved features to %s rows=%d", FEATURES_CSV, len(df))
    return FEATURES_CSV

# --- Model training (quick baseline RF) ---
def train_quick_rf(features_csv):
    log.info("Training quick RandomForest on %s", features_csv)
    df = pd.read_csv(features_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    feature_cols = ["ret","ema8","ema21","ema50","ema200","rsi14","atr14","vol10","mom5","ema_diff"]
    df["target"] = (df["Close"].pct_change().shift(-1) > 0).astype(int)
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    X = df[feature_cols]; y = df["target"]
    # train on all (quick baseline)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    log.info("Saved RF model to %s", MODEL_PATH)
    return MODEL_PATH

# --- Inference & signal generation ---
def infer_and_write_signals():
    # ensure features exist
    if not FEATURES_CSV.exists():
        # find a CSV to build features from
        candidate = None
        if DATA_DIR.exists():
            csvs = sorted([p for p in DATA_DIR.glob("*.csv")], key=lambda p: p.name)
            if csvs:
                candidate = csvs[-1]
        if candidate is None and DATA_PRIMARY:
            candidate = Path(DATA_PRIMARY)
        if candidate is None or not candidate.exists():
            log.error("No input CSV found in data/ to compute features. Place a TradingView CSV in data/ and retry.")
            return {"error":"no_input_csv"}
        compute_features_from_primary(candidate)

    # ensure model exists
    if not MODEL_PATH.exists():
        train_quick_rf(FEATURES_CSV)

    # run inference on last row
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    last = df.iloc[-1]
    model = joblib.load(MODEL_PATH)
    feature_cols = ["ret","ema8","ema21","ema50","ema200","rsi14","atr14","vol10","mom5","ema_diff"]
    X_last = last[feature_cols].values.reshape(1, -1)
    prob = float(model.predict_proba(X_last)[0, 1])
    # trend checks
    weekly_ok = None
    weekly_csv = DATA_DIR / "XAU_USD_Historical_Data_weekly.csv"
    if weekly_csv.exists():
        try:
            wk = safe_read(weekly_csv)
            wk["ema21"] = wk["Close"].ewm(span=21).mean()
            wk["ema200"] = wk["Close"].ewm(span=200).mean()
            weekly_ok = bool(wk["ema21"].iloc[-1] > wk["ema200"].iloc[-1])
        except Exception:
            weekly_ok = None

    conservative_entry = float(last["ema21"] + 0.5 * last["atr14"])
    aggressive_entry = float(last["ema21"] + 1.5 * last["atr14"])
    safer_entry = float(last["ema21"] + 0.25 * last["atr14"])

    signal = bool((prob > 0.55) and (last["ema21"] > last["ema200"]))

    out = {
        "timestamp": str(datetime.utcnow()),
        "bar_timestamp": str(last["timestamp"]),
        "close": float(last["Close"]),
        "probability": prob,
        "signal": "BUY" if signal else "NO_LONG",
        "conservative_entry": conservative_entry,
        "aggressive_entry": aggressive_entry,
        "safer_entry": safer_entry,
        "weekly_ok": weekly_ok
    }

    with open(SIGNALS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote signals.json: %s", out)
    return out

# --- Background scheduler ---
_stop_event = threading.Event()
def scheduler_loop(interval_secs=INFER_INTERVAL_SECS):
    log.info("Scheduler loop started, interval %s seconds", interval_secs)
    while not _stop_event.is_set():
        try:
            infer_and_write_signals()
        except Exception as e:
            log.exception("Error during scheduled inference: %s", e)
        # wait with early exit
        for _ in range(int(interval_secs / 10)):
            if _stop_event.is_set():
                break
            time.sleep(10)

# --- Flask app ---
app = Flask("xau_agent")

@app.route("/ping")
def ping():
    return jsonify({"ping": "pong", "time": str(datetime.utcnow())})

@app.route("/")
def root():
    return jsonify({"status":"ok", "time": str(datetime.utcnow())})

@app.route("/run", methods=["POST","GET"])
def run_once():
    try:
        out = infer_and_write_signals()
        return jsonify({"status":"ok", "result": out})
    except Exception as e:
        log.exception("Error on /run")
        return jsonify({"status":"error","message": str(e)}), 500

# Start background thread and Flask app
def start_service():
    # start scheduler thread
    t = threading.Thread(target=scheduler_loop, args=(INFER_INTERVAL_SECS,), daemon=True)
    t.start()
    # run flask (host 0.0.0.0 so Render/VM can access)
    port = int(os.getenv("PORT", 10000))
    log.info("Starting Flask on 0.0.0.0:%s", port)
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    try:
        start_service()
    except KeyboardInterrupt:
        _stop_event.set()
        log.info("Shutting down...")
