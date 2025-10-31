# main.py — XAU/USD AI Agent (Investing.com via RapidAPI + AlphaVantage hybrid)
from flask import Flask, jsonify
import pandas as pd, numpy as np, joblib, json, threading, time, os, requests
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# === CONFIGURATION ===
# ======================================================
app = Flask(__name__)
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"
MODEL_DAY = ROOT / "model_day.pkl"
SCALER_DAY = ROOT / "scaler_day.pkl"
MODEL_HR = ROOT / "model_hr.pkl"
SCALER_HR = ROOT / "scaler_hr.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))  # hourly retrain
PORT = int(os.getenv("PORT", 10000))
SELF_PING_URL = os.getenv("SELF_PING_URL", None)
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "demo")      # from Render env
ALPHAV_API_KEY = os.getenv("ALPHAV_API_KEY", "demo")  # from Render env

# New feature flags / params (env override)
VP_BINS = int(os.getenv("VP_BINS", 24))                      # volume profile bins
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", 3))             # candles to examine for FVG
CONFIRMATION_CANDLES = int(os.getenv("CONFIRMATION_CANDLES", 1))  # confirmation requirement
CONFIRMATION_TYPE = os.getenv("CONFIRMATION_TYPE", "trend")  # "trend", "close_above", "close_below"

# ======================================================
# === INDICATOR ENGINE ===
# ======================================================
def compute_indicators(df):
    """Compute key technical indicators."""
    df = df.copy()
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["mom5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()

    df = df.dropna().reset_index(drop=True)
    return df

# ======================================================
# === VOLUME PROFILE CONTEXT ===
# ======================================================
def compute_vp_profile(df, bins=VP_BINS):
    """
    Compute a simple volume-by-price profile.
    Returns DataFrame with columns ['price', 'volume'] aggregated by bins,
    and the heavyweight price (HVN) and low-volume node (LVN).
    If Volume column missing, returns None.
    """
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return None

    prices = df["Close"]
    vols = df["Volume"]

    # Build bins across price range
    price_min, price_max = prices.min(), prices.max()
    if price_max <= price_min:
        return None

    bins_edges = np.linspace(price_min, price_max, bins + 1)
    bin_idx = np.digitize(prices, bins_edges) - 1
    bin_vol = {}
    for i, v in zip(bin_idx, vols.fillna(0)):
        if 0 <= i < bins:
            bin_vol.setdefault(i, 0.0)
            bin_vol[i] += float(v)
    vp = []
    for i in range(bins):
        center = (bins_edges[i] + bins_edges[i + 1]) / 2.0
        vp.append({"bin": i, "price": center, "volume": bin_vol.get(i, 0.0)})
    vp_df = pd.DataFrame(vp).sort_values("volume", ascending=False).reset_index(drop=True)

    # Heavy and low volume nodes
    hvn = vp_df.iloc[0].to_dict() if not vp_df.empty else None
    lvn = vp_df.tail(1).iloc[0].to_dict() if not vp_df.empty else None
    return {"vp": vp_df, "hvn": hvn, "lvn": lvn, "bins_edges": bins_edges}

def last_price_in_low_volume_node(df, vp_profile, width_pct=0.005):
    """
    Return True if last close is near a low-volume node (favors breakouts),
    False if inside HVN (choppy), or None if no VP data.
    width_pct is tolerance relative to price (e.g., 0.5%).
    """
    if vp_profile is None:
        return None
    last_price = float(df["Close"].iloc[-1])
    lvn = vp_profile.get("lvn")
    hvn = vp_profile.get("hvn")
    if lvn:
        if abs(last_price - lvn["price"]) / last_price <= width_pct:
            return True
    if hvn:
        if abs(last_price - hvn["price"]) / last_price <= width_pct:
            return False
    return None

# ======================================================
# === FAIR VALUE GAP (FVG) DETECTION ===
# ======================================================
def detect_fvg(df, lookback=FVG_LOOKBACK):
    """
    Detect simple Fair Value Gaps within the last `lookback` candles.
    Simple rule (common retail definition):
      - Bullish FVG: candle i's low > candle i-1's high (upward gap)
      - Bearish FVG: candle i's high < candle i-1's low (downward gap)
    Returns dict with 'type': 'bullish'/'bearish'/None and 'gap_price_range'
    """
    if len(df) < 2:
        return {"type": None, "range": None}

    # Inspect last `lookback` transitions
    recent = df.tail(lookback + 1).reset_index(drop=True)
    for i in range(1, len(recent)):
        prev = recent.loc[i - 1]
        cur = recent.loc[i]
        # bullish gap (price jumped up leaving a gap)
        if cur["Low"] > prev["High"]:
            return {"type": "bullish", "range": (prev["High"], cur["Low"])}
        # bearish gap (price dropped)
        if cur["High"] < prev["Low"]:
            return {"type": "bearish", "range": (cur["High"], prev["Low"])}
    return {"type": None, "range": None}

# ======================================================
# === SMART DELAY / CONFIRMATION ===
# ======================================================
def check_confirmation(df, direction, confirmation_candles=CONFIRMATION_CANDLES, confirmation_type=CONFIRMATION_TYPE):
    """
    direction: 'BUY' or 'SELL'
    confirmation_type:
      - 'trend': require last N closes to be increasing (BUY) or decreasing (SELL)
      - 'close_above': require last candle close > last ema8 (BUY) or close < ema8 (SELL)
      - 'close_above_ma21' etc could be extended
    Returns True if confirmed, False otherwise.
    """
    if len(df) < max(confirmation_candles, 2) + 1:
        return False

    recent = df.tail(confirmation_candles + 1).reset_index(drop=True)
    if confirmation_type == "trend":
        if direction == "BUY":
            # Each subsequent close should be >= previous close (allow equality)
            return all(recent.loc[i, "Close"] >= recent.loc[i - 1, "Close"] for i in range(1, len(recent)))
        else:  # SELL
            return all(recent.loc[i, "Close"] <= recent.loc[i - 1, "Close"] for i in range(1, len(recent)))
    elif confirmation_type == "close_above":
        # require latest close > ema8 for BUY, < ema8 for SELL
        last = recent.iloc[-1]
        if "ema8" not in df.columns or np.isnan(last["ema8"]):
            return False
        if direction == "BUY":
            return last["Close"] > last["ema8"]
        else:
            return last["Close"] < last["ema8"]
    else:
        # default to trend
        return check_confirmation(df, direction, confirmation_candles, "trend")

# ======================================================
# === DATA FETCHING (Investing + Alpha) ===
# ======================================================
def fetch_investing_daily():
    """Fetch daily XAU/USD data from Investing.com via RapidAPI."""
    print("📥 Fetching daily XAU/USD data (RapidAPI)...")
    url = "https://investing-com.p.rapidapi.com/price/historical"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "investing-com.p.rapidapi.com"
    }
    params = {"symbol": "XAU/USD", "interval": "1d", "from": "2000-01-01"}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
        data = res.json()
        df = pd.DataFrame(data["data"])
        df["Date"] = pd.to_datetime(df["date"])
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume":"Volume"}, inplace=True)
        # If Investing doesn't return Volume, that's fine — our code handles it.
        df = df[["Date", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])].sort_values("Date")
        df.to_csv(DAILY_FILE, index=False)
        print(f"✅ Saved daily data → {DAILY_FILE} ({len(df)} rows)")
        return df
    except Exception as e:
        print("❌ RapidAPI fetch error:", e)
        if DAILY_FILE.exists():
            print("⚠️ Using cached daily data.")
            return pd.read_csv(DAILY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

def fetch_alpha_hourly():
    """Fetch hourly XAU/USD data from Alpha Vantage."""
    print("📥 Fetching hourly XAU/USD data (Alpha Vantage)...")
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY"
        f"&from_symbol=XAU&to_symbol=USD&interval=60min"
        f"&apikey={ALPHAV_API_KEY}&outputsize=full"
    )

    try:
        r = requests.get(url, timeout=15)
        data = r.json().get("Time Series FX (60min)", {})
        if not data:
            raise ValueError("Empty hourly dataset from Alpha Vantage.")
        df = pd.DataFrame(data).T
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.astype(float)
        df["Date"] = pd.to_datetime(df.index)
        df = df.sort_values("Date")
        # AlphaVantage FX_INTRADAY doesn't provide volume — OK.
        df.to_csv(HOURLY_FILE, index=False)
        print(f"✅ Saved hourly data → {HOURLY_FILE} ({len(df)} rows)")
        return df
    except Exception as e:
        print("❌ AlphaVantage error:", e)
        if HOURLY_FILE.exists():
            print("⚠️ Using cached hourly data.")
            return pd.read_csv(HOURLY_FILE, parse_dates=["Date"])
        return pd.DataFrame()

# ======================================================
# === MODEL + SIGNAL ===
# ======================================================
def train_model(X, y, model_path):
    """Train RandomForest with StandardScaler."""
    if len(X) < 50:
        print("⚠️ Not enough samples to train model.")
        return None
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)

    joblib.dump(model, model_path)
    joblib.dump(scaler, str(model_path).replace(".pkl", "_scaler.pkl"))
    print(f"🤖 Model saved {model_path.name} (val acc={acc:.3f})")
    return model

def apply_context_filters(df, raw_signal):
    """
    Apply Volume Profile, FVG, and Smart Delay confirmation.
    raw_signal: dict with 'signal' (BUY/SELL) and 'confidence'
    Returns dict with possibly updated 'signal' (N/A if filtered out) and annotations.
    """
    annotations = {}
    sig = raw_signal.get("signal", "N/A")
    if sig not in ("BUY", "SELL"):
        return {**raw_signal, "filtered": True, "reason": "invalid_raw_signal", **annotations}

    # Compute indicators if not present
    if "ema8" not in df.columns or df.isna().any().any():
        try:
            df = compute_indicators(df)
        except Exception:
            pass

    # Volume profile
    vp = compute_vp_profile(df)
    vp_context = last_price_in_low_volume_node(df, vp)
    annotations["vp_context"] = vp_context  # True -> in LVN (good for breakout); False -> in HVN (choppy)

    # Fair Value Gap
    fvg = detect_fvg(df)
    annotations["fvg"] = fvg

    # If FVG exists, prefer its direction; if opposite to model, lower confidence or block
    if fvg["type"] == "bullish" and sig == "SELL":
        # model disagrees with gap direction; attenuate
        annotations["fvg_conflict"] = True
        # If strong conflict and no LVN support, filter out
        if vp_context is False or vp_context is None:
            return {**raw_signal, "signal": "N/A", "confidence": 0, "filtered": True, "reason": "fvg_conflict_hvn", **annotations}
    elif fvg["type"] == "bearish" and sig == "BUY":
        annotations["fvg_conflict"] = True
        if vp_context is False or vp_context is None:
            return {**raw_signal, "signal": "N/A", "confidence": 0, "filtered": True, "reason": "fvg_conflict_hvn", **annotations}

    # Smart delay / confirmation
    confirmed = check_confirmation(df, sig)
    annotations["confirmed"] = confirmed
    if not confirmed:
        # mark as filtered — we require confirmation before releasing signal
        return {**raw_signal, "signal": "N/A", "confidence": 0, "filtered": True, "reason": "no_confirmation", **annotations}

    # If we've reached here, signal is allowed
    return {**raw_signal, "filtered": False, "reason": None, **annotations}

def generate_signal(df, model_path, label):
    """Compute indicators, train model, and output signal (with context filters)."""
    if df.empty:
        return {"label": label, "timestamp": str(datetime.now(timezone.utc)), "signal": "N/A", "confidence": 0}

    df = compute_indicators(df)
    # binary next-close-up target
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["ema8", "ema21", "ema50", "atr14", "rsi14", "vol10", "mom5"]
    df = df.dropna(subset=features + ["target"])
    if df.empty:
        print(f"⚠️ No valid rows for {label}")
        return {"label": label, "timestamp": str(datetime.now(timezone.utc)), "signal": "N/A", "confidence": 0}

    model = train_model(df[features], df["target"], model_path)
    if model is None:
        return {"label": label, "timestamp": str(datetime.now(timezone.utc)), "signal": "N/A", "confidence": 0}

    scaler_path = str(model_path).replace(".pkl", "_scaler.pkl")
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        print("❗ Scaler not found; skipping prediction")
        return {"label": label, "timestamp": str(datetime.now(timezone.utc)), "signal": "N/A", "confidence": 0}

    last = df[features].iloc[[-1]]
    prob = float(model.predict_proba(scaler.transform(last))[0][1])
    pred = int(model.predict(scaler.transform(last))[0])
    signal = "BUY" if pred == 1 else "SELL"

    raw = {
        "label": label,
        "timestamp": str(datetime.now(timezone.utc)),
        "signal": signal,
        "confidence": round(prob * 100, 2)
    }

    # Apply volume-profile, FVG and smart-delay confirmation filters
    filtered = apply_context_filters(df, raw)
    return filtered

# ======================================================
# === SIGNAL PIPELINE ===
# ======================================================
def build_train_and_signal():
    """Fetch, train, and produce both daily + hourly signals."""
    daily_df = fetch_investing_daily()
    hr_df = fetch_alpha_hourly()

    day_sig = generate_signal(daily_df, MODEL_DAY, "Daily")
    hr_sig = generate_signal(hr_df, MODEL_HR, "Hourly")

    combined = {
        "timestamp": str(datetime.now(timezone.utc)),
        "daily": day_sig,
        "hourly": hr_sig
    }

    json.dump(combined, open(SIGNALS_FILE, "w"), indent=2)
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
        except Exception:
            history = []
    history.append(combined)
    json.dump(history[-100:], open(HISTORY_FILE, "w"), indent=2)

    print(f"[{combined['timestamp']}] 🕐 Hourly:{hr_sig['signal']}({hr_sig.get('confidence',0)}%) | 📅 Daily:{day_sig['signal']}({day_sig.get('confidence',0)}%)")
    return combined

# ======================================================
# === BACKGROUND REFRESH LOOP ===
# ======================================================
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

# ======================================================
# === FLASK ROUTES ===
# ======================================================
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
        current = {"daily": {"signal": "N/A", "confidence": 0},
                   "hourly": {"signal": "N/A", "confidence": 0}}

    color_day = "#0f0" if current["daily"]["signal"] == "BUY" else "#f55"
    color_hr = "#0f0" if current["hourly"]["signal"] == "BUY" else "#f55"

    return f"""
    <html><head><title>XAU/USD AI Dashboard</title><meta http-equiv="refresh" content="600"></head>
    <body style="font-family:Arial;background:#0d1117;color:#fff;text-align:center;padding:40px;">
    <h1>XAU/USD Dual-Timeframe AI Agent</h1>
    <div style="display:inline-block;background:#161b22;padding:20px;border-radius:12px;">
      <h2>📅 Daily: <span style="color:{color_day}">{current['daily']['signal']}</span> ({current['daily'].get('confidence',0)}%)</h2>
      <h2>🕐 Hourly: <span style="color:{color_hr}">{current['hourly']['signal']}</span> ({current['hourly'].get('confidence',0)}%)</h2>
      <p><button onclick="fetch('/signal').then(()=>location.reload())">🔄 Refresh Signal</button></p>
      <p>Last Update: {current.get('timestamp','N/A')}</p>
    </div>
    </body></html>
    """

# ======================================================
# === START SERVER ===
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    print(f"🚀 Starting Flask on port {PORT} | Refresh every {REFRESH_INTERVAL_SECS}s (RapidAPI + AlphaVantage)")
    app.run(host="0.0.0.0", port=PORT)
