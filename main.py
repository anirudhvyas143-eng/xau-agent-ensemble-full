# main.py — XAU/USD AI Agent 
# Includes: 5-key AlphaVantage rotation (hardcoded), Finnhub fallback,twelvedata fallback
# indicators, ensemble ML, fusion, SL/TP, backtest, Optuna, RL stub, Flask API.

import os
import time
import json
import random
import threading
import math
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional libs
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import backtrader as bt
    HAS_BACKTRADER = True
except Exception:
    HAS_BACKTRADER = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    HAS_RL = True
except Exception:
    HAS_RL = False

# Finnhub (fallback) - optional import, handled gracefully if not installed
try:
    import finnhub
    HAS_FINNHUB = True
except Exception:
    HAS_FINNHUB = False

from flask import Flask, jsonify, request

# -----------------------
# CONFIG & FILES
# -----------------------
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"
MODEL_PATH = ROOT / "ensemble_model.pkl"
SCALER_PATH = ROOT / "ensemble_scaler.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

# === Hardcoded AlphaVantage API keys (exact values you gave) ===
ALPHAV_API_KEYS = [
    "XWZFB7RP8I4SWCMZ",  # key A
    "XUU2PYO481XBYWR4",  # key B
    "94CMKYJJQUVN51AT",  # key C
    "0DZCC9GW6YJBNUYP",  # key D
    "I2SOZBI81ZWMY56L",  # key E
]
ALPHAV_API_KEY = random.choice(ALPHAV_API_KEYS)

# === Finnhub fallback key (you provided this) ===
# If you want to rotate or change, edit this value.
FINNHUB_KEY = "d449o49r01qge0d1701gd449o49r01qge0d17020"

# Symbol: prefer XAU for FX endpoints; fallback to GLD (ETF)
# For AlphaVantage FX endpoints we use from_symbol/to_symbol.
SYMBOL_FX = ("XAU", "USD")
SYMBOL_EQ = "GLD"  # fallback ETF symbol

REFRESH_INTERVAL_SECS = int(os.getenv("REFRESH_INTERVAL_SECS", 3600))  # default 1 hour (you used 3600 in logs)
PORT = int(os.getenv("PORT", 10000))

# Feature flags / params
VP_BINS = int(os.getenv("VP_BINS", 24))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", 3))
CONFIRMATION_CANDLES = int(os.getenv("CONFIRMATION_CANDLES", 1))
CONFIRMATION_TYPE = os.getenv("CONFIRMATION_TYPE", "trend")  # "trend" or "close_above"

# Risk sizing
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000.0))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% default
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", 0.0005))   # 5 bps default

# -----------------------
# Helper: AlphaVantage request with rotation
# -----------------------
def try_alpha_request(params):
    """Call AlphaVantage rotating through provided keys. Return JSON or {}."""
    global ALPHAV_API_KEY
    base = "https://www.alphavantage.co/query"
    for key in ALPHAV_API_KEYS:
        params["apikey"] = key
        try:
            r = requests.get(base, params=params, timeout=25)
            data = r.json()
        except Exception as e:
            print(f"⚠️ Network/error with key ...{key[-4:]}: {e}")
            continue

        msg = json.dumps(data).lower()
        # detect "info" or limit messages
        if "rate limit" in msg or "thank you for using alpha vantage" in msg or "invalid api call" in msg or "note" in msg:
            print(f"⚠️ Key ending ...{key[-4:]} returned info/limit message; rotating.")
            continue

        ALPHAV_API_KEY = key
        print(f"✅ Using AlphaVantage key ending ...{key[-4:]}")
        return data
    print("❌ All AlphaVantage keys exhausted or returned limit/info message.")
    return {}

# -----------------------
# Finnhub fallback helpers
# -----------------------
def fetch_from_finnhub_fx(symbol="OANDA:XAU_USD", resolution="D", count=2000):
    """
    Fetch candles from Finnhub for gold forex (OANDA:XAU_USD).
    resolution: "1","5","15","30","60","D","W","M"
    count: approximate number of bars to request (used to compute from timestamp)
    """
    if not HAS_FINNHUB:
        print("⚠️ Finnhub library not available.")
        return pd.DataFrame()
    try:
        client = finnhub.Client(api_key=FINNHUB_KEY)
        now = int(time.time())
        # resolution mapping: if resolution is 'D' use days, else treat as minutes/hours.
        if resolution.upper() == "D":
            # count days -> seconds
            _from = now - count * 24 * 3600
        elif resolution.upper() == "W":
            _from = now - count * 7 * 24 * 3600
        else:
            # assume minutes/hours - use 3600 seconds per bar for hourly
            # If resolution is '60' treat as hourly
            step_seconds = 60 if resolution == "1" else (60 * int(resolution))
            _from = now - count * step_seconds
        res = client.forex_candles(symbol, resolution, _from, now)
        if res and res.get("s") == "ok":
            df = pd.DataFrame({
                "Date": pd.to_datetime(res["t"], unit="s"),
                "Open": res["o"],
                "High": res["h"],
                "Low": res["l"],
                "Close": res["c"],
                "Volume": res.get("v", [0] * len(res["t"]))
            })
            df = df.sort_values("Date").reset_index(drop=True)
            print(f"✅ Finnhub FX fallback returned {len(df)} rows for {symbol} @ {resolution}")
            return df
        else:
            print("⚠️ Finnhub FX returned no data or error:", res)
            return pd.DataFrame()
    except Exception as e:
        print("❌ Finnhub FX fetch error:", e)
        return pd.DataFrame()

def fetch_from_finnhub_symbol(symbol="GLD", resolution="D", count=2000):
    """
    If Finnhub supports the ETF ticker in your plan, try that (less likely).
    Here as a generic fallback — primarily use FX fetch above.
    """
    # For safety, reuse FX fetch fallback; keep this as a stub.
    return pd.DataFrame()
    # =========================================================
# ✅ TwelveData fallback (final safety layer)
# =========================================================
from twelvedata import TDClient

TWELVEDATA_KEY = "daf266a898fd450caed947b15cfba53e"

def fetch_from_twelvedata(symbol="XAU/USD", interval="1min"):
    """
    Fetch latest XAU/USD price or candle using TwelveData SDK.
    Triggered only if both AlphaVantage & Finnhub fail.
    """
    try:
        td = TDClient(apikey=TWELVEDATA_KEY)

        # Try latest price endpoint first
        price_data = td.price(symbol=symbol).as_json()
        if "price" in price_data:
            price = float(price_data["price"])
            print(f"✅ TwelveData (latest) → {price}")
            df = pd.DataFrame([{
                "Date": pd.Timestamp.utcnow(),
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": 0
            }])
            return df

        # Fallback: use 1-min time series
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=1).as_json()
        if isinstance(ts, list) and len(ts) > 0 and "close" in ts[0]:
            close = float(ts[0]["close"])
            print(f"✅ TwelveData (timeseries) → {close}")
            df = pd.DataFrame([{
                "Date": ts[0]["datetime"],
                "Open": ts[0]["open"],
                "High": ts[0]["high"],
                "Low": ts[0]["low"],
                "Close": ts[0]["close"],
                "Volume": 0
            }])
            return df

        print("⚠️ TwelveData returned unexpected format:", ts)
    except Exception as e:
        print("❌ TwelveData fetch error:", e)

    print("🚫 TwelveData failed.")
    return pd.DataFrame()

# -----------------------
# Data fetchers (AlphaVantage primary; Finnhub fallback
# -----------------------
def fetch_fx_daily_xauusd():
    """Use FX_DAILY from AlphaVantage for XAU/USD (preferred)."""
    print("📥 Fetching XAU/USD daily via FX_DAILY (AlphaVantage)...")
    params = {
        "function": "FX_DAILY",
        "from_symbol": SYMBOL_FX[0],
        "to_symbol": SYMBOL_FX[1],
        "outputsize": "full",
    }
    data = try_alpha_request(params)
    if not data or "Time Series FX (Daily)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
    df = df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"})
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.to_csv(DAILY_FILE, index=False)
    print(f"✅ Saved FX daily → {DAILY_FILE} ({len(df)} rows)")
    return df, data

def fetch_fx_intraday_xauusd():
    """Use FX_INTRADAY for hourly XAU/USD from AlphaVantage."""
    print("📥 Fetching XAU/USD hourly via FX_INTRADAY 60min (AlphaVantage)...")
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": SYMBOL_FX[0],
        "to_symbol": SYMBOL_FX[1],
        "interval": "60min",
        "outputsize": "compact",
    }
    data = try_alpha_request(params)
    if not data or "Time Series FX (60min)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series FX (60min)"], orient="index")
    df = df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"})
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.to_csv(HOURLY_FILE, index=False)
    print(f"✅ Saved FX hourly → {HOURLY_FILE} ({len(df)} rows)")
    return df, data

def fetch_symbol_daily_globaleq(symbol=SYMBOL_EQ):
    """Fallback: use TIME_SERIES_DAILY for GLD ETF (AlphaVantage)."""
    print(f"📥 Fetching {symbol} daily via TIME_SERIES_DAILY (AlphaVantage fallback)...")
    params = {"function":"TIME_SERIES_DAILY","symbol":symbol,"outputsize":"full"}
    data = try_alpha_request(params)
    if not data or "Time Series (Daily)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"})
    df.index.name="Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    fn = DAILY_FILE.parent / f"{symbol}_daily.csv"
    df.to_csv(fn, index=False)
    print(f"✅ Saved fallback daily → {fn} ({len(df)} rows)")
    return df, data

def fetch_symbol_intraday_globaleq(symbol=SYMBOL_EQ):
    """Fallback: use TIME_SERIES_INTRADAY for GLD ETF hourly (AlphaVantage)."""
    print(f"📥 Fetching {symbol} hourly via TIME_SERIES_INTRADAY (AlphaVantage fallback)...")
    params = {"function":"TIME_SERIES_INTRADAY","symbol":symbol,"interval":"60min","outputsize":"compact"}
    data = try_alpha_request(params)
    if not data or "Time Series (60min)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
    df = df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"})
    df.index.name="Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    fn = HOURLY_FILE.parent / f"{symbol}_hourly.csv"
    df.to_csv(fn, index=False)
    print(f"✅ Saved fallback hourly → {fn} ({len(df)} rows)")
    return df, data

def fetch_daily():
    """Try AlphaVantage FX daily first; fallback to AlphaVantage symbol; then Finnhub."""
    df, data = fetch_fx_daily_xauusd()
    if df.empty:
        df, data = fetch_symbol_daily_globaleq(SYMBOL_EQ)
    # If still empty and Finnhub available, try Finnhub daily (D)
    if df.empty:
        if df.empty:
    print("⚠️ AlphaVantage + Finnhub failed — trying TwelveData fallback.")
    df = fetch_from_twelvedata()

def fetch_hourly():
    """Try AlphaVantage FX intraday first; fallback to AlphaVantage symbol intraday; then Finnhub hourly."""
    df, data = fetch_fx_intraday_xauusd()
    if df.empty:
        df, data = fetch_symbol_intraday_globaleq(SYMBOL_EQ)
    # If still empty and Finnhub available, try Finnhub hourly ('60')
    if df.empty:
        if df.empty:
    print("⚠️ AlphaVantage + Finnhub failed — trying TwelveData fallback.")
    df = fetch_from_twelvedata()

# -----------------------
# Indicators & features
# -----------------------
def compute_rsi(series, periods=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(periods).mean()
    avg_loss = loss.rolling(periods).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(df):
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # EMAs
    df["ema8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # MACD
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # RSI
    df["rsi14"] = compute_rsi(df["Close"], 14)
    df["mom5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()
    df = df.dropna().reset_index(drop=True)
    return df

# -----------------------
# Volume Profile
# -----------------------
def compute_vp_profile(df, bins=VP_BINS):
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return None
    prices = df["Close"].values
    volumes = df["Volume"].fillna(0).astype(float).values
    pmin, pmax = prices.min(), prices.max()
    if pmax <= pmin:
        return None
    edges = np.linspace(pmin, pmax, bins+1)
    idx = np.digitize(prices, edges) - 1
    bin_vol = np.zeros(bins)
    for i, v in zip(idx, volumes):
        if 0 <= i < bins:
            bin_vol[i] += v
    centers = (edges[:-1] + edges[1:]) / 2.0
    vp_df = pd.DataFrame({"price": centers, "volume": bin_vol}).sort_values("volume", ascending=False).reset_index(drop=True)
    hvn = vp_df.iloc[0].to_dict() if not vp_df.empty else None
    lvn = vp_df.iloc[-1].to_dict() if not vp_df.empty else None
    return {"vp": vp_df, "hvn": hvn, "lvn": lvn, "edges": edges}

def last_price_in_lvn(df, vp_profile, width_pct=0.005):
    if vp_profile is None:
        return None
    last_price = float(df["Close"].iloc[-1])
    lvn = vp_profile.get("lvn")
    hvn = vp_profile.get("hvn")
    if lvn and abs(last_price - lvn["price"]) / last_price <= width_pct:
        return True
    if hvn and abs(last_price - hvn["price"]) / last_price <= width_pct:
        return False
    return None

# -----------------------
# Fair Value Gap
# -----------------------
def detect_fvg(df, lookback=FVG_LOOKBACK):
    if len(df) < 2:
        return {"type": None, "range": None}
    recent = df.tail(lookback + 1).reset_index(drop=True)
    for i in range(1, len(recent)):
        prev = recent.loc[i-1]; cur = recent.loc[i]
        if cur["Low"] > prev["High"]:
            return {"type": "bullish", "range": (prev["High"], cur["Low"])}
        if cur["High"] < prev["Low"]:
            return {"type": "bearish", "range": (cur["High"], prev["Low"])}
    return {"type": None, "range": None}

# -----------------------
# Regime detection
# -----------------------
def regime_detector(df):
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
    sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["Close"].mean()
    trend = df["Close"].iloc[-1] > sma50
    if vol < 0.005 and trend: return "low-vol-trend"
    if vol < 0.005 and not trend: return "low-vol-range"
    if vol >= 0.005: return "high-vol"
    return "unknown"

# -----------------------
# SL/TP & sizing
# -----------------------
def calc_sl_tp(latest_row, side="BUY", atr_mult_sl=1.5, atr_mult_tp=2.5):
    entry = float(latest_row["Close"])
    atr = float(latest_row.get("atr14", 0.0) or 0.0)
    if atr <= 0:
        return {"entry": round(entry,4), "sl": None, "tp": None}
    if side == "BUY":
        sl = entry - atr * atr_mult_sl
        tp = entry + atr * atr_mult_tp
    else:
        sl = entry + atr * atr_mult_sl
        tp = entry - atr * atr_mult_tp
    return {"entry": round(entry,4), "sl": round(sl,4), "tp": round(tp,4)}

def position_size(entry, sl, account_size=ACCOUNT_SIZE, risk_per_trade=RISK_PER_TRADE):
    if sl is None:
        return 0
    risk_amount = account_size * risk_per_trade
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0
    qty = math.floor(risk_amount / risk_per_unit)
    return max(0, int(qty))

def apply_slippage(price, side="BUY", bps=SLIPPAGE_BPS):
    if price is None:
        return None
    if side == "BUY":
        return price * (1 + bps)
    return price * (1 - bps)

# -----------------------
# Ensemble training & predict
# -----------------------
def train_ensemble(daily_df):
    df = compute_indicators(daily_df.rename(columns={"Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}))
    if df.empty or len(df) < 80:
        print("⚠️ Not enough data to train.")
        return None
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    features = ["ema8","ema21","ema50","atr14","rsi14","mom5","vol10"]
    X = df[features]; y = df["target"]
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(Xs, y)
    lgbm = None
    if HAS_LGB:
        try:
            lgbm = lgb.LGBMClassifier(n_estimators=200)
            lgbm.fit(X, y)
        except Exception as e:
            print("⚠️ LightGBM train failed:", e)
            lgbm = None
    joblib.dump({"rf": rf, "lgbm": lgbm, "features": features}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Ensemble trained & saved.")
    return {"rf": rf, "lgbm": lgbm, "features": features}

def ensemble_predict_proba(df_latest):
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    ens = joblib.load(MODEL_PATH); scaler = joblib.load(SCALER_PATH)
    features = ens["features"]
    X = df_latest[features].iloc[[-1]]
    Xs = scaler.transform(X)
    probs = []
    if ens.get("rf"):
        probs.append(float(ens["rf"].predict_proba(Xs)[0][1]))
    if ens.get("lgbm"):
        try:
            probs.append(float(ens["lgbm"].predict_proba(X)[0][1]))
        except Exception:
            pass
    if not probs:
        return None
    return sum(probs) / len(probs)

# -----------------------
# Fusion / voting
# -----------------------
def fuse_signal(model_prob, df):
    last = df.iloc[-1]; annotations={}
    macd_pos = last["macd"] > last["macd_signal"]
    ema_trend = last["ema8"] > last["ema21"] > last["ema50"]
    rsi = float(last["rsi14"])
    vp = compute_vp_profile(df); vp_ok = last_price_in_lvn(df, vp)
    fvg = detect_fvg(df)
    annotations.update({"macd_pos":bool(macd_pos),"ema_trend":bool(ema_trend),"rsi14":rsi,"vp_context":vp_ok,"fvg":fvg})
    vote = 0
    if model_prob is not None:
        vote += 1 if model_prob > 0.6 else -1 if model_prob < 0.4 else 0
    vote += 1 if macd_pos else -1
    vote += 1 if ema_trend else -1
    if rsi < 30: vote += 1
    elif rsi > 70: vote -= 1
    if fvg["type"] == "bullish": vote += 1
    elif fvg["type"] == "bearish": vote -= 1
    if vp_ok is False: vote -= 1
    elif vp_ok is True: vote += 1
    if vote >= 2: signal = "BUY"
    elif vote <= -2: signal = "SELL"
    else: signal = "N/A"
    conf = 0
    if model_prob is not None:
        conf = int(min(95, max(20, 50 + (model_prob - 0.5) * 100)))
    conf += int(vote * 5)
    conf = int(max(0, min(99, conf)))
    return {"signal": signal, "confidence": conf, "annotations": annotations, "vote": vote}

# -----------------------
# Backtest simulator
# -----------------------
def simulate_backtest(df, entry_side="BUY", entry_index=None, sl=None, tp=None, qty=1, slippage_bps=SLIPPAGE_BPS):
    if entry_index is None:
        entry_index = len(df)-1
    entry_price = float(df.iloc[entry_index]["Close"])
    entry_price = apply_slippage(entry_price, side=entry_side, bps=slippage_bps)
    sl_price = sl; tp_price = tp
    for i in range(entry_index+1, len(df)):
        low = float(df.iloc[i]["Low"])
        high = float(df.iloc[i]["High"])
        if entry_side == "BUY":
            if sl_price is not None and low <= sl_price:
                exit_price = apply_slippage(sl_price, side="SELL", bps=slippage_bps)
                pnl = (exit_price - entry_price) * qty
                return {"pnl": pnl, "exit_price": exit_price, "index": i, "reason": "SL"}
            if tp_price is not None and high >= tp_price:
                exit_price = apply_slippage(tp_price, side="SELL", bps=slippage_bps)
                pnl = (exit_price - entry_price) * qty
                return {"pnl": pnl, "exit_price": exit_price, "index": i, "reason": "TP"}
        else:
            if sl_price is not None and high >= sl_price:
                exit_price = apply_slippage(sl_price, side="BUY", bps=slippage_bps)
                pnl = (entry_price - exit_price) * qty
                return {"pnl": pnl, "exit_price": exit_price, "index": i, "reason": "SL"}
            if tp_price is not None and low <= tp_price:
                exit_price = apply_slippage(tp_price, side="BUY", bps=slippage_bps)
                pnl = (entry_price - exit_price) * qty
                return {"pnl": pnl, "exit_price": exit_price, "index": i, "reason": "TP"}
    last_price = apply_slippage(float(df.iloc[-1]["Close"]), side="SELL" if entry_side=="BUY" else "BUY", bps=slippage_bps)
    pnl = (last_price - entry_price) * qty if entry_side == "BUY" else (entry_price - last_price) * qty
    return {"pnl": pnl, "exit_price": last_price, "index": len(df)-1, "reason": "HOLD"}

# -----------------------
# Pipeline: build_train_and_signal
# -----------------------
def build_train_and_signal():
    daily_df = fetch_daily()
    hourly_df = fetch_hourly()
    day_out = {"signal":"N/A","confidence":0}
    hr_out = {"signal":"N/A","confidence":0}
    if not daily_df.empty:
        try:
            proc = compute_indicators(daily_df.rename(columns={"Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}))
            if not MODEL_PATH.exists() or (time.time() - MODEL_PATH.stat().st_mtime) > 86400*3:
                train_ensemble(daily_df)
            model_prob = ensemble_predict_proba(proc)
            fused = fuse_signal(model_prob, proc)
            sltp = calc_sl_tp(proc.iloc[-1], side=fused["signal"] if fused["signal"] in ("BUY","SELL") else "BUY")
            qty = position_size(sltp["entry"], sltp["sl"])
            day_out = {**fused, **sltp, "qty": qty}
        except Exception as e:
            print("⚠️ Day pipeline error:", e)
    if not hourly_df.empty:
        try:
            proc_h = compute_indicators(hourly_df.rename(columns={"Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}))
            model_prob_h = ensemble_predict_proba(proc_h) or (ensemble_predict_proba(proc) if not daily_df.empty else None)
            fused_h = fuse_signal(model_prob_h, proc_h if not proc_h.empty else proc)
            sltp_h = calc_sl_tp(proc_h.iloc[-1] if not proc_h.empty else proc.iloc[-1], side=fused_h["signal"] if fused_h["signal"] in ("BUY","SELL") else "BUY")
            qty_h = position_size(sltp_h["entry"], sltp_h["sl"])
            hr_out = {**fused_h, **sltp_h, "qty": qty_h}
        except Exception as e:
            print("⚠️ Hour pipeline error:", e)
    combined = {"timestamp": datetime.now(timezone.utc).isoformat(), "daily": day_out, "hourly": hr_out}
    with open(SIGNALS_FILE, "w") as f:
        json.dump(combined, f, indent=2)
    history = []
    if HISTORY_FILE.exists():
        try: history = json.load(open(HISTORY_FILE))
        except Exception: history = []
    history.append(combined)
    json.dump(history[-200:], open(HISTORY_FILE, "w"), indent=2)
    print(f"[{combined['timestamp']}] Hourly:{hr_out['signal']}({hr_out.get('confidence',0)}%) Daily:{day_out['signal']}({day_out.get('confidence',0)}%)")
    return combined

# -----------------------
# Optuna (optional)
# -----------------------
def run_optuna_study(n_trials=20):
    if not HAS_OPTUNA:
        print("⚠️ Optuna not installed.")
        return None
    daily_df = fetch_daily()
    if daily_df.empty:
        print("⚠️ No daily data for Optuna.")
        return None
    proc = compute_indicators(daily_df)
    proc["target"] = (proc["Close"].shift(-1) > proc["Close"]).astype(int)
    proc.dropna(inplace=True)
    features = ["ema8","ema21","ema50","atr14","rsi14","mom5","vol10"]
    X = proc[features]; y = proc["target"]
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 400)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(rf, X, y, cv=3, scoring="accuracy")
        return 1.0 - score.mean()
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    print("Optuna best params:", study.best_params)
    return study.best_params

# -----------------------
# RL stub (optional)
# -----------------------
if HAS_RL:
    class SimpleTradingEnv(gym.Env):
        metadata = {"render.modes":["human"]}
        def __init__(self, df):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.index = 0
            self.action_space = gym.spaces.Discrete(3)  # 0:flat,1:long,2:short
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32)
        def reset(self):
            self.index = 0
            return self._get_obs(), {}
        def _get_obs(self):
            row = self.df.iloc[self.index].fillna(0).values.astype(np.float32)
            return row
        def step(self, action):
            cur = float(self.df.iloc[self.index]["Close"])
            self.index = min(self.index + 1, len(self.df)-1)
            nxt = float(self.df.iloc[self.index]["Close"])
            ret = (nxt - cur) / cur
            pos = 0
            if action==1: pos = 1
            elif action==2: pos = -1
            reward = pos * ret
            done = (self.index >= len(self.df)-1)
            return self._get_obs(), reward, done, False, {}
        def render(self, mode="human"):
            pass
    def train_rl_agent(n_steps=50000):
        daily_df = fetch_daily()
        if daily_df.empty:
            print("⚠️ No daily data to train RL.")
            return None
        proc = compute_indicators(daily_df)
        env = SimpleTradingEnv(proc[["Open","High","Low","Close","atr14","rsi14","macd"]])
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        model.learn(total_timesteps=n_steps)
        model.save("ppo_agent")
        print("✅ RL agent trained & saved.")
        return model

# -----------------------
# Flask app & routes
# -----------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "active_api_key_tail": ALPHAV_API_KEY[-4:],
        "note": "AlphaVantage primary; Finnhub fallback (if available)."
    })

@app.route("/signal")
def signal_route():
    """
    Returns the latest signal JSON. Contains:
      - daily: { signal, confidence, entry, sl, tp, qty, annotations }
      - hourly: same
    """
    if SIGNALS_FILE.exists():
        try:
            return jsonify(json.load(open(SIGNALS_FILE)))
        except Exception:
            pass
    return jsonify(build_train_and_signal())

@app.route("/signal_url")
def signal_url():
    """Return the absolute URL to the /signal endpoint for linking from logs/dashboards."""
    base = request.host_url.rstrip("/")
    return jsonify({"signal_url": f"{base}/signal"})

@app.route("/history")
def history_route():
    if HISTORY_FILE.exists():
        try:
            return jsonify(json.load(open(HISTORY_FILE)))
        except Exception:
            pass
    return jsonify([])

@app.route("/predict")
def predict_route():
    if not DAILY_FILE.exists():
        return jsonify({"error": "no daily file"})
    df = pd.read_csv(DAILY_FILE)
    proc = compute_indicators(df.rename(columns={"Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}))
    prob = ensemble_predict_proba(proc)
    return jsonify({"proba_up": prob})

@app.route("/backtest", methods=["POST"])
def backtest_route():
    payload = request.get_json() or {}
    tf = payload.get("tf", "daily")
    start = payload.get("from", None)
    end = payload.get("to", None)
    if tf == "hourly":
        df = fetch_hourly()
    else:
        df = fetch_daily()
    if df.empty: return jsonify({"error": "no data"})
    df = df.copy()
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df.index)
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    if start: df = df[df["Date"] >= pd.to_datetime(start)]
    if end: df = df[df["Date"] <= pd.to_datetime(end)]
    proc = compute_indicators(df)
    if proc.empty: return jsonify({"error":"no processed data"})
    # simple walk-forward simulated trades
    results = []
    window = 200
    for i in range(window, len(proc)-10, 10):
        train = proc.iloc[:i]
        test = proc.iloc[i:i+10]
        try:
            X = train[["ema8","ema21","ema50","atr14","rsi14","mom5","vol10"]]
            y = (train["Close"].shift(-1) > train["Close"]).astype(int).shift(-1).fillna(0)
            y = y[:-1]; X = X[:-1]
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            last = test.iloc[0:1]
            pred = rf.predict(last[["ema8","ema21","ema50","atr14","rsi14","mom5","vol10"]])[0]
            side = "BUY" if pred == 1 else "SELL"
            sltp = calc_sl_tp(last.iloc[0], side=side)
            qty = max(1, position_size(sltp["entry"], sltp["sl"]))
            sim = simulate_backtest(pd.concat([last, test]), entry_side=side, entry_index=0, sl=sltp["sl"], tp=sltp["tp"], qty=qty)
            results.append(sim["pnl"])
        except Exception:
            continue
    if not results:
        return jsonify({"error": "no backtest trades"})
    arr = np.array(results)
    return jsonify({"trades": len(arr), "total_pnl": float(arr.sum()), "avg_pnl": float(arr.mean()), "winrate": float((arr>0).sum()/len(arr))})

@app.route("/train_optuna", methods=["POST"])
def train_optuna_route():
    if not HAS_OPTUNA:
        return jsonify({"error":"optuna not installed"})
    n = int(request.get_json().get("n_trials", 20))
    params = run_optuna_study(n_trials=n)
    return jsonify({"best_params": params})

@app.route("/train_rl", methods=["POST"])
def train_rl_route():
    if not HAS_RL:
        return jsonify({"error":"stable-baselines3 or gym not installed"})
    n_steps = int(request.get_json().get("n_steps", 50000))
    res = train_rl_agent(n_steps=n_steps)
    return jsonify({"status":"trained" if res else "failed"})

# -----------------------
# Background refresh
# -----------------------
def background_loop():
    while True:
        try:
            build_train_and_signal()
        except Exception as e:
            print("Background loop error:", e)
        time.sleep(REFRESH_INTERVAL_SECS)

# -----------------------
# Start server
# -----------------------
if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    print(f"🚀 Starting Flask on port {PORT} | Refresh every {REFRESH_INTERVAL_SECS}s (AlphaVantage keys rotated, Finnhub fallback enabled)")
    app.run(host="0.0.0.0", port=PORT)
