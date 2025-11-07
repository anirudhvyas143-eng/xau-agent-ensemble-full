# main.py — XAU/USD AI Agent
# Includes: 5-key AlphaVantage rotation (hardcoded) + TwelveData fallback
# Indicators, ensemble ML, fusion, SL/TP, backtest, Optuna, RL stub, Flask API.

import os, time, json, random, threading, math
from datetime import datetime, timezone
from pathlib import Path
import requests, pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import time
app = Flask(__name__)

# optional libs
try: import lightgbm as lgb; HAS_LGB=True
except: HAS_LGB=False
try: import optuna; HAS_OPTUNA=True
except: HAS_OPTUNA=False
try: import backtrader as bt; HAS_BACKTRADER=True
except: HAS_BACKTRADER=False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    HAS_RL=True
except: HAS_RL=False

# ---------------- CONFIG ----------------
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(exist_ok=True)
DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"
MODEL_PATH = ROOT / "ensemble_model.pkl"
SCALER_PATH = ROOT / "ensemble_scaler.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

ALPHAV_API_KEYS = [
    "XWZFB7RP8I4SWCMZ",
    "XUU2PYO481XBYWR4",
    "94CMKYJJQUVN51AT",
    "0DZCC9GW6YJBNUYP",
    "I2SOZBI81ZWMY56L",
]
ALPHAV_API_KEY = random.choice(ALPHAV_API_KEYS)

# TwelveData key + SDK import (SDK optional — we fallback to REST)
try:
    from twelvedata import TDClient
    HAS_TWELVE_SDK = True
except Exception:
    HAS_TWELVE_SDK = False
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "daf266a898fd450caed947b15cfba53e")

SYMBOL_FX = ("XAU", "USD")
SYMBOL_EQ = "GLD"


PORT = int(os.getenv("PORT", 10000))
VP_BINS = int(os.getenv("VP_BINS", 24))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", 3))
CONFIRMATION_CANDLES = int(os.getenv("CONFIRMATION_CANDLES", 1))
CONFIRMATION_TYPE = os.getenv("CONFIRMATION_TYPE", "trend")
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000.0))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", 0.0005))

# ---------------- AlphaVantage rotation ----------------
def try_alpha_request(params):
    base = "https://www.alphavantage.co/query"
    for key in ALPHAV_API_KEYS:
        params["apikey"] = key
        try:
            r = requests.get(base, params=params, timeout=25)
            data = r.json()
        except Exception as e:
            print(f"⚠️ AlphaVantage error {key[-4:]}: {e}")
            continue
        msg = json.dumps(data).lower()
        if any(k in msg for k in ["rate limit", "thank you", "invalid", "note"]):
            print(f"⚠️ Key ...{key[-4:]} hit limit.")
            continue
        print(f"✅ Using AlphaVantage key ...{key[-4:]}")
        return data
    print("❌ All AlphaVantage keys exhausted.")
    return {}

# ---------------- TwelveData fallback (robust) ----------------
def fetch_from_twelvedata(symbol="XAU/USD", interval="1day", outputsize=30):
    """
    Try SDK -> REST fallback. Returns DataFrame or empty DataFrame.
    Handles different return shapes from SDK and REST.
    """
    # 1) Try SDK (if available)
    if HAS_TWELVE_SDK:
        try:
            td = TDClient(apikey=TWELVEDATA_KEY)
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize).as_json()
            # SDK may return dict with "values" or a list
            if isinstance(ts, dict):
                # common SDK format: {'values': [...], 'status': 'ok', ...}
                vals = ts.get("values") or ts.get("data") or ts.get("result")
                if isinstance(vals, list) and len(vals) > 0:
                    df = pd.DataFrame(vals)
                    # unify names
                    rename_map = {c: c.capitalize() for c in df.columns if c in ["datetime","open","high","low","close","volume"]}
                    df.rename(columns=rename_map, inplace=True)
                    if "Datetime" in df.columns:
                        df.rename(columns={"Datetime":"Date"}, inplace=True)
                    if "Date" not in df.columns and "datetime" in df.columns:
                        df.rename(columns={"datetime":"Date"}, inplace=True)
                    for c in ["Open","High","Low","Close","Volume"]:
                        if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.sort_values("Date").reset_index(drop=True)
                        if "Close" in df.columns:
                            print(f"✅ TwelveData(SDK) returned {len(df)} rows.")
                            return df
            elif isinstance(ts, list) and len(ts) > 0:
                df = pd.DataFrame(ts)
                for c in ["open","high","low","close"]: 
                    if c in df.columns:
                        df[c]=pd.to_numeric(df[c], errors="coerce")
                df["datetime"]=pd.to_datetime(df["datetime"])
                df.rename(columns={"datetime":"Date","open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
                df["Volume"]=df.get("volume",0)
                df=df.sort_values("Date").reset_index(drop=True)
                print(f"✅ TwelveData(SDK) returned {len(df)} rows.")
                return df
        except Exception as e:
            print("❌ TwelveData SDK fetch error:", e)

    # 2) REST fallback (public API)
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVEDATA_KEY,
        }
        r = requests.get(url, params=params, timeout=25)
        j = r.json()
        # Expected: {'meta': {...}, 'values': [...], 'status': 'ok'}
        vals = j.get("values") or j.get("data") or j.get("values", None)
        if isinstance(vals, list) and len(vals) > 0:
            df = pd.DataFrame(vals)
            # unify names
            if "datetime" in df.columns:
                df["Date"] = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"])
            for c in ["open","high","low","close","volume"]:
                if c in df.columns:
                    df[c.capitalize()] = pd.to_numeric(df[c], errors="coerce")
            # keep standard column names if available
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in df.columns and c.lower() in df.columns:
                    df[c] = pd.to_numeric(df[c.lower()], errors="coerce")
            if "Date" in df.columns and "Close" in df.columns:
                df = df[["Date","Open","High","Low","Close"] + ([c for c in ["Volume"] if c in df.columns])]
                df = df.sort_values("Date").reset_index(drop=True)
                print(f"✅ TwelveData(REST) returned {len(df)} rows.")
                return df
        # If error field present, log it
        if "message" in j:
            print("❌ TwelveData REST message:", j.get("message"))
        else:
            print("❌ TwelveData REST returned unexpected payload:", j)
    except Exception as e:
        print("❌ TwelveData REST fetch error:", e)

    return pd.DataFrame()

# ---------------- Data fetchers ----------------
def fetch_fx_daily_xauusd():
    print("📥 Fetching XAU/USD daily via FX_DAILY (AlphaVantage)...")
    params = {"function":"FX_DAILY","from_symbol":SYMBOL_FX[0],"to_symbol":SYMBOL_FX[1],"outputsize":"full"}
    data = try_alpha_request(params)
    if not data or "Time Series FX (Daily)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date"); df.to_csv(DAILY_FILE, index=False)
    print(f"✅ Saved FX daily ({len(df)})")
    return df, data

def fetch_fx_intraday_xauusd():
    print("📥 Fetching XAU/USD hourly via FX_INTRADAY 60min...")
    params={"function":"FX_INTRADAY","from_symbol":SYMBOL_FX[0],"to_symbol":SYMBOL_FX[1],"interval":"60min","outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series FX (60min)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series FX (60min)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date"); df.to_csv(HOURLY_FILE, index=False)
    print(f"✅ Saved FX hourly ({len(df)})")
    return df, data

def fetch_symbol_daily_globaleq(symbol=SYMBOL_EQ):
    print(f"📥 Fetching {symbol} daily via TIME_SERIES_DAILY...")
    params={"function":"TIME_SERIES_DAILY","symbol":symbol,"outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series (Daily)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date")
    fn=DAILY_FILE.parent/f"{symbol}_daily.csv"; df.to_csv(fn,index=False)
    print(f"✅ Saved fallback daily → {fn} ({len(df)})")
    return df, data

def fetch_symbol_intraday_globaleq(symbol=SYMBOL_EQ):
    print(f"📥 Fetching {symbol} hourly via TIME_SERIES_INTRADAY...")
    params={"function":"TIME_SERIES_INTRADAY","symbol":symbol,"interval":"60min","outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series (60min)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date")
    fn=HOURLY_FILE.parent/f"{symbol}_hourly.csv"; df.to_csv(fn,index=False)
    print(f"✅ Saved fallback hourly → {fn} ({len(df)})")
    return df
    # ---------------------------------------------------------------------------------
# Extended TwelveData Hourly Fetcher (up to 35000 records using batching)
# ---------------------------------------------------------------------------------
def fetch_twelvedata_xauusd(api_key, interval="1h", total_records=100000):
    """
    Fetch extended XAU/USD hourly data from TwelveData beyond 5,000-record limit
    by batching multiple 5,000-record requests and merging results.
    Works safely on TwelveData free (Trial) plan.
    """
    print(f"📡 Fetching up to {total_records} rows of {interval} data from TwelveData...")

    base_url = "https://api.twelvedata.com/time_series"
    end_date = datetime.utcnow()
    all_data = []
    batch_size = 5000
    batches = total_records // batch_size  # e.g., 100000 / 5000 = 20

    # ~208 days per batch for 1h data
    step_days = 5000 / 24 if interval == "1h" else 5000

    for i in range(batches):
        start_date = end_date - timedelta(days=step_days)
        params = {
            "symbol": "XAU/USD",
            "interval": interval,
            "apikey": api_key,
            "outputsize": batch_size,
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data = json_data.get("values", [])

            if not data:
                print(f"⚠️ Batch {i+1}: No more data available or API limit reached.")
                break

            all_data.extend(data)
            print(f"✅ Batch {i+1}/{batches} complete — total {len(all_data)} rows collected.")
            end_date = start_date  # move backward in time
            time.sleep(8)  # safe delay for TwelveData free plan (8 req/min)

        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            break

    if not all_data:
        print("❌ No data fetched from TwelveData.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    print(f"🎯 Final merged dataset: {len(df)} rows of {interval} XAU/USD data.")
    return df


# ---------------------------------------------------------------------------------
# Extended TwelveData Daily Fetcher (up to 35000daily candles)
# ---------------------------------------------------------------------------------
def fetch_twelvedata_xauusd_daily(api_key, total_records=100000):
    """
    Fetch extended XAU/USD daily candles from TwelveData beyond 5,000-record limit.
    Works safely even on TwelveData free plan.
    """
    print(f"📡 Fetching up to {total_records} rows of DAILY data from TwelveData...")

    base_url = "https://api.twelvedata.com/time_series"
    end_date = datetime.utcnow()
    all_data = []
    batch_size = 5000
    batches = total_records // batch_size  # 13 batches

    step_days = 5000  # each batch = ~5000 days

    for i in range(batches):
        start_date = end_date - timedelta(days=step_days)
        params = {
            "symbol": "XAU/USD",
            "interval": "1day",
            "apikey": api_key,
            "outputsize": batch_size,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data = json_data.get("values", [])

            if not data:
                print(f"⚠️ Batch {i+1}: No more data available or API limit reached.")
                break

            all_data.extend(data)
            print(f"✅ Batch {i+1}/{batches} complete — total {len(all_data)} rows collected.")
            end_date = start_date
            time.sleep(8)  # ⏳ safe delay for free plan (8 req/min)

        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            break

    if not all_data:
        print("❌ No daily data fetched from TwelveData.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    print(f"🎯 Final merged dataset: {len(df)} DAILY candles of XAU/USD.")
    return df
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
# (rest of your functions: volume profile, fvg, regime, sl/tp, ensemble etc.)
# I'll reuse your previously defined helpers (position_size, calc_sl_tp, train_ensemble, ensemble_predict_proba, fuse_signal, simulate_backtest, build_train_and_signal)
# For brevity, include them unchanged — paste in your version or reuse the earlier implementation.
# (Below I include them verbatim from your provided code so the file is self-contained)
# -----------------------

# Volume profile
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

def regime_detector(df):
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
    sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["Close"].mean()
    trend = df["Close"].iloc[-1] > sma50
    if vol < 0.005 and trend: return "low-vol-trend"
    if vol < 0.005 and not trend: return "low-vol-range"
    if vol >= 0.005: return "high-vol"
    return "unknown"

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
# Build, train, and signal generator (with offline fallback)
# -----------------------
def build_train_and_signal():
    print("📊 Running build_train_and_signal() refresh...")

    # === DAILY DATA FETCH ===
    try:
        daily_df, _ = fetch_fx_daily_xauusd()
        if daily_df.empty:
            print("⚠️ AlphaVantage DAILY failed — trying TwelveData fallback.")
            daily_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1day", total_records=12000)
        if daily_df.empty:
            print("⚠️ Daily TwelveData also failed — will use cached file if available.")
        else:
            daily_df.to_csv("daily.csv", index=False)
            print(f"✅ DAILY data saved locally: {len(daily_df)} rows")
    except Exception as e:
        print("❌ Daily data fetch error:", e)
        daily_df = pd.read_csv("daily.csv") if os.path.exists("daily.csv") else pd.DataFrame()

    # === HOURLY DATA FETCH ===
    try:
        hourly_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1h", total_records=35000)
        if hourly_df.empty:
            print("⚠️ Hourly TwelveData returned no data — will use cached file if available.")
        else:
            hourly_df.to_csv("hourly.csv", index=False)
            print(f"✅ HOURLY data saved locally: {len(hourly_df)} rows")
    except Exception as e:
        print("❌ Hourly data fetch error:", e)
        hourly_df = pd.read_csv("hourly.csv") if os.path.exists("hourly.csv") else pd.DataFrame()

    # === SIGNAL GENERATION PIPELINE ===
    day_out = {"signal": "N/A", "confidence": 0}
    hr_out = {"signal": "N/A", "confidence": 0}

    # === DAILY MODEL PIPELINE ===
    if not daily_df.empty:
        try:
            daily_df.rename(columns=lambda x: x.capitalize(), inplace=True)
            proc = compute_indicators(daily_df)

            # Retrain if model older than 3 days
            if not MODEL_PATH.exists() or (time.time() - MODEL_PATH.stat().st_mtime) > 86400 * 3:
                print("🧠 Retraining ensemble model (daily)...")
                train_ensemble(daily_df)

            model_prob = ensemble_predict_proba(proc)
            fused = fuse_signal(model_prob, proc)
            sltp = calc_sl_tp(proc.iloc[-1], side=fused["signal"] if fused["signal"] in ("BUY", "SELL") else "BUY")
            qty = position_size(sltp["entry"], sltp["sl"])
            day_out = {**fused, **sltp, "qty": qty}
        except Exception as e:
            print("⚠️ Daily pipeline error:", e)

    # === HOURLY MODEL PIPELINE ===
    if not hourly_df.empty:
        try:
            hourly_df.rename(columns=lambda x: x.capitalize(), inplace=True)
            proc_h = compute_indicators(hourly_df)
            model_prob_h = ensemble_predict_proba(proc_h)
            fused_h = fuse_signal(model_prob_h, proc_h)
            sltp_h = calc_sl_tp(proc_h.iloc[-1], side=fused_h["signal"] if fused_h["signal"] in ("BUY", "SELL") else "BUY")
            qty_h = position_size(sltp_h["entry"], sltp_h["sl"])
            hr_out = {**fused_h, **sltp_h, "qty": qty_h}
        except Exception as e:
            print("⚠️ Hourly pipeline error:", e)

    # === OFFLINE FALLBACK (when both APIs failed) ===
    if daily_df.empty and hourly_df.empty:
        print("⚡ Using offline cached data and saved ensemble model for signal regeneration...")
        if os.path.exists("ensemble_model.pkl"):
            with open("ensemble_model.pkl", "rb") as f:
                ensemble = pickle.load(f)
            if os.path.exists("daily.csv") and os.path.exists("hourly.csv"):
                daily_df = pd.read_csv("daily.csv")
                hourly_df = pd.read_csv("hourly.csv")

                proc_d = compute_indicators(daily_df)
                prob_d = ensemble_predict_proba(proc_d)
                fused_d = fuse_signal(prob_d, proc_d)

                proc_h = compute_indicators(hourly_df)
                prob_h = ensemble_predict_proba(proc_h)
                fused_h = fuse_signal(prob_h, proc_h)

                day_out = fused_d
                hr_out = fused_h
                print(f"✅ Offline signal regeneration complete — Daily:{day_out['signal']} | Hourly:{hr_out['signal']}")
            else:
                print("⚠️ No cached CSV data found for offline regeneration.")
        else:
            print("⚠️ No saved ensemble model found — cannot regenerate offline signals.")

    # === SAVE & LOG RESULTS ===
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "daily": day_out,
        "hourly": hr_out
    }

    with open(SIGNALS_FILE, "w") as f:
        json.dump(combined, f, indent=2)

    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
        except Exception:
            history = []
    history.append(combined)
    json.dump(history[-200:], open(HISTORY_FILE, "w"), indent=2)

    print(f"[{combined['timestamp']}] Hourly:{hr_out['signal']} ({hr_out.get('confidence', 0)}%) | "
          f"Daily:{day_out['signal']} ({day_out.get('confidence', 0)}%)")

    return combined

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "active_api_key_tail": ALPHAV_API_KEY[-4:],
        "note": "AlphaVantage primary; TwelveData fallback enabled."
    })

@app.route("/signal")
def signal_route():
    # return saved file if exists; else build on demand
    if SIGNALS_FILE.exists():
        try:
            return jsonify(json.load(open(SIGNALS_FILE)))
        except Exception:
            pass
    # build fresh (this will call fetch_daily/fetch_hourly)
    out = build_train_and_signal()
    return jsonify(out)

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
@app.route("/signal/refresh", methods=["POST"])

        # Load your saved CSVs
        daily = pd.read_csv("daily.csv")
        hourly = pd.read_csv("hourly.csv")

        # Generate signals
        daily_signal = generate_ensemble_signal(daily, timeframe="daily")
        hourly_signal = generate_ensemble_signal(hourly, timeframe="hourly")

        # Save to signals.json
        import json, datetime
        signal_data = {
            "daily": daily_signal,
            "hourly": hourly_signal,
            "time": datetime.datetime.utcnow().isoformat()
        }
        with open("signals.json", "w") as f:
            json.dump(signal_data, f, indent=2)

        return jsonify({"status": "✅ Signal regenerated", "data": signal_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
     # Force daily-only signal generation
from build import build_train_and_signal, generate_signals_daily_only

# Instead of calling full build_train_and_signal(), call:
generate_signals_daily_only()   
# -----------------------
import pandas as pd
import joblib  # or pickle depending on how you save your model

def generate_signals_daily_only():
    # Load daily CSV
    daily_df = pd.read_csv("daily.csv")  # your saved daily data

    # Load trained daily ensemble
    model = joblib.load("ensemble_daily.pkl")  # adjust filename if different

    # Prepare features (example, adjust to your features)
    X = daily_df[['feature1','feature2','feature3','feature4','feature5','feature6','feature7']]

    # Predict
    daily_df['signal'] = model.predict(X)

    # Save signals
    daily_df.to_csv("daily_signals.csv", index=False)
    print("✅ Daily signals generated and saved to daily_signals.csv")
# Background refresh (run once)
# -----------------------
def background_loop():
    try:
        print("🚀 Running one-time build and signal generation...")
        build_train_and_signal()
        print("✅ Signal generation complete — check /signal endpoint.")
    except Exception as e:
        print("Background loop error:", e)
        # main.py — XAU/USD AI Agent
# Includes: 5-key AlphaVantage rotation (hardcoded) + TwelveData fallback
# Indicators, ensemble ML, fusion, SL/TP, backtest, Optuna, RL stub, Flask API.

import os, time, json, random, threading, math
from datetime import datetime, timezone
from pathlib import Path
import requests, pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import time
app = Flask(__name__)

# optional libs
try: import lightgbm as lgb; HAS_LGB=True
except: HAS_LGB=False
try: import optuna; HAS_OPTUNA=True
except: HAS_OPTUNA=False
try: import backtrader as bt; HAS_BACKTRADER=True
except: HAS_BACKTRADER=False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    HAS_RL=True
except: HAS_RL=False

# ---------------- CONFIG ----------------
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(exist_ok=True)
DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"
MODEL_PATH = ROOT / "ensemble_model.pkl"
SCALER_PATH = ROOT / "ensemble_scaler.pkl"
SIGNALS_FILE = ROOT / "signals.json"
HISTORY_FILE = ROOT / "signals_history.json"

ALPHAV_API_KEYS = [
    "XWZFB7RP8I4SWCMZ",
    "XUU2PYO481XBYWR4",
    "94CMKYJJQUVN51AT",
    "0DZCC9GW6YJBNUYP",
    "I2SOZBI81ZWMY56L",
]
ALPHAV_API_KEY = random.choice(ALPHAV_API_KEYS)

# TwelveData key + SDK import (SDK optional — we fallback to REST)
try:
    from twelvedata import TDClient
    HAS_TWELVE_SDK = True
except Exception:
    HAS_TWELVE_SDK = False
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "daf266a898fd450caed947b15cfba53e")

SYMBOL_FX = ("XAU", "USD")
SYMBOL_EQ = "GLD"


# -------------------------
# Auto-refresh loop (every 24h)
# -------------------------
while True:
    try:
        print("🔄 Auto-refreshing signals (daily)...")
        refresh_signals()
        print("✅ Daily signal refresh completed. Waiting 24h...")
    except Exception as e:
        print(f"⚠️ Auto-refresh error: {e}")
    time.sleep(86400)  # wait 24 hours before next refresh
PORT = int(os.getenv("PORT", 10000))
VP_BINS = int(os.getenv("VP_BINS", 24))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", 3))
CONFIRMATION_CANDLES = int(os.getenv("CONFIRMATION_CANDLES", 1))
CONFIRMATION_TYPE = os.getenv("CONFIRMATION_TYPE", "trend")
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000.0))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", 0.0005))

# ---------------- AlphaVantage rotation ----------------
def try_alpha_request(params):
    base = "https://www.alphavantage.co/query"
    for key in ALPHAV_API_KEYS:
        params["apikey"] = key
        try:
            r = requests.get(base, params=params, timeout=25)
            data = r.json()
        except Exception as e:
            print(f"⚠️ AlphaVantage error {key[-4:]}: {e}")
            continue
        msg = json.dumps(data).lower()
        if any(k in msg for k in ["rate limit", "thank you", "invalid", "note"]):
            print(f"⚠️ Key ...{key[-4:]} hit limit.")
            continue
        print(f"✅ Using AlphaVantage key ...{key[-4:]}")
        return data
    print("❌ All AlphaVantage keys exhausted.")
    return {}

# ---------------- TwelveData fallback (robust) ----------------
def fetch_from_twelvedata(symbol="XAU/USD", interval="1day", outputsize=30):
    """
    Try SDK -> REST fallback. Returns DataFrame or empty DataFrame.
    Handles different return shapes from SDK and REST.
    """
    # 1) Try SDK (if available)
    if HAS_TWELVE_SDK:
        try:
            td = TDClient(apikey=TWELVEDATA_KEY)
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize).as_json()
            # SDK may return dict with "values" or a list
            if isinstance(ts, dict):
                # common SDK format: {'values': [...], 'status': 'ok', ...}
                vals = ts.get("values") or ts.get("data") or ts.get("result")
                if isinstance(vals, list) and len(vals) > 0:
                    df = pd.DataFrame(vals)
                    # unify names
                    rename_map = {c: c.capitalize() for c in df.columns if c in ["datetime","open","high","low","close","volume"]}
                    df.rename(columns=rename_map, inplace=True)
                    if "Datetime" in df.columns:
                        df.rename(columns={"Datetime":"Date"}, inplace=True)
                    if "Date" not in df.columns and "datetime" in df.columns:
                        df.rename(columns={"datetime":"Date"}, inplace=True)
                    for c in ["Open","High","Low","Close","Volume"]:
                        if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.sort_values("Date").reset_index(drop=True)
                        if "Close" in df.columns:
                            print(f"✅ TwelveData(SDK) returned {len(df)} rows.")
                            return df
            elif isinstance(ts, list) and len(ts) > 0:
                df = pd.DataFrame(ts)
                for c in ["open","high","low","close"]: 
                    if c in df.columns:
                        df[c]=pd.to_numeric(df[c], errors="coerce")
                df["datetime"]=pd.to_datetime(df["datetime"])
                df.rename(columns={"datetime":"Date","open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
                df["Volume"]=df.get("volume",0)
                df=df.sort_values("Date").reset_index(drop=True)
                print(f"✅ TwelveData(SDK) returned {len(df)} rows.")
                return df
        except Exception as e:
            print("❌ TwelveData SDK fetch error:", e)

    # 2) REST fallback (public API)
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVEDATA_KEY,
        }
        r = requests.get(url, params=params, timeout=25)
        j = r.json()
        # Expected: {'meta': {...}, 'values': [...], 'status': 'ok'}
        vals = j.get("values") or j.get("data") or j.get("values", None)
        if isinstance(vals, list) and len(vals) > 0:
            df = pd.DataFrame(vals)
            # unify names
            if "datetime" in df.columns:
                df["Date"] = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"])
            for c in ["open","high","low","close","volume"]:
                if c in df.columns:
                    df[c.capitalize()] = pd.to_numeric(df[c], errors="coerce")
            # keep standard column names if available
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in df.columns and c.lower() in df.columns:
                    df[c] = pd.to_numeric(df[c.lower()], errors="coerce")
            if "Date" in df.columns and "Close" in df.columns:
                df = df[["Date","Open","High","Low","Close"] + ([c for c in ["Volume"] if c in df.columns])]
                df = df.sort_values("Date").reset_index(drop=True)
                print(f"✅ TwelveData(REST) returned {len(df)} rows.")
                return df
        # If error field present, log it
        if "message" in j:
            print("❌ TwelveData REST message:", j.get("message"))
        else:
            print("❌ TwelveData REST returned unexpected payload:", j)
    except Exception as e:
        print("❌ TwelveData REST fetch error:", e)

    return pd.DataFrame()

# ---------------- Data fetchers ----------------
def fetch_fx_daily_xauusd():
    print("📥 Fetching XAU/USD daily via FX_DAILY (AlphaVantage)...")
    params = {"function":"FX_DAILY","from_symbol":SYMBOL_FX[0],"to_symbol":SYMBOL_FX[1],"outputsize":"full"}
    data = try_alpha_request(params)
    if not data or "Time Series FX (Daily)" not in data:
        return pd.DataFrame(), data
    df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date"); df.to_csv(DAILY_FILE, index=False)
    print(f"✅ Saved FX daily ({len(df)})")
    return df, data

def fetch_fx_intraday_xauusd():
    print("📥 Fetching XAU/USD hourly via FX_INTRADAY 60min...")
    params={"function":"FX_INTRADAY","from_symbol":SYMBOL_FX[0],"to_symbol":SYMBOL_FX[1],"interval":"60min","outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series FX (60min)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series FX (60min)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date"); df.to_csv(HOURLY_FILE, index=False)
    print(f"✅ Saved FX hourly ({len(df)})")
    return df, data

def fetch_symbol_daily_globaleq(symbol=SYMBOL_EQ):
    print(f"📥 Fetching {symbol} daily via TIME_SERIES_DAILY...")
    params={"function":"TIME_SERIES_DAILY","symbol":symbol,"outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series (Daily)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date")
    fn=DAILY_FILE.parent/f"{symbol}_daily.csv"; df.to_csv(fn,index=False)
    print(f"✅ Saved fallback daily → {fn} ({len(df)})")
    return df, data

def fetch_symbol_intraday_globaleq(symbol=SYMBOL_EQ):
    print(f"📥 Fetching {symbol} hourly via TIME_SERIES_INTRADAY...")
    params={"function":"TIME_SERIES_INTRADAY","symbol":symbol,"interval":"60min","outputsize":"full"}
    data=try_alpha_request(params)
    if not data or "Time Series (60min)" not in data: return pd.DataFrame(), data
    df=pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
    df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"}, inplace=True)
    df.index.name="Date"; df=df.reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date")
    fn=HOURLY_FILE.parent/f"{symbol}_hourly.csv"; df.to_csv(fn,index=False)
    print(f"✅ Saved fallback hourly → {fn} ({len(df)})")
    return df
    # ---------------------------------------------------------------------------------
# Extended TwelveData Hourly Fetcher (up to 35000 records using batching)
# ---------------------------------------------------------------------------------
def fetch_twelvedata_xauusd(api_key, interval="1h", total_records=100000):
    """
    Fetch extended XAU/USD hourly data from TwelveData beyond 5,000-record limit
    by batching multiple 5,000-record requests and merging results.
    Works safely on TwelveData free (Trial) plan.
    """
    print(f"📡 Fetching up to {total_records} rows of {interval} data from TwelveData...")

    base_url = "https://api.twelvedata.com/time_series"
    end_date = datetime.utcnow()
    all_data = []
    batch_size = 5000
    batches = total_records // batch_size  # e.g., 100000 / 5000 = 20

    # ~208 days per batch for 1h data
    step_days = 5000 / 24 if interval == "1h" else 5000

    for i in range(batches):
        start_date = end_date - timedelta(days=step_days)
        params = {
            "symbol": "XAU/USD",
            "interval": interval,
            "apikey": api_key,
            "outputsize": batch_size,
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data = json_data.get("values", [])

            if not data:
                print(f"⚠️ Batch {i+1}: No more data available or API limit reached.")
                break

            all_data.extend(data)
            print(f"✅ Batch {i+1}/{batches} complete — total {len(all_data)} rows collected.")
            end_date = start_date  # move backward in time
            time.sleep(8)  # safe delay for TwelveData free plan (8 req/min)

        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            break

    if not all_data:
        print("❌ No data fetched from TwelveData.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    print(f"🎯 Final merged dataset: {len(df)} rows of {interval} XAU/USD data.")
    return df


# ---------------------------------------------------------------------------------
# Extended TwelveData Daily Fetcher (up to 35000daily candles)
# ---------------------------------------------------------------------------------
def fetch_twelvedata_xauusd_daily(api_key, total_records=100000):
    """
    Fetch extended XAU/USD daily candles from TwelveData beyond 5,000-record limit.
    Works safely even on TwelveData free plan.
    """
    print(f"📡 Fetching up to {total_records} rows of DAILY data from TwelveData...")

    base_url = "https://api.twelvedata.com/time_series"
    end_date = datetime.utcnow()
    all_data = []
    batch_size = 5000
    batches = total_records // batch_size  # 13 batches

    step_days = 5000  # each batch = ~5000 days

    for i in range(batches):
        start_date = end_date - timedelta(days=step_days)
        params = {
            "symbol": "XAU/USD",
            "interval": "1day",
            "apikey": api_key,
            "outputsize": batch_size,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data = json_data.get("values", [])

            if not data:
                print(f"⚠️ Batch {i+1}: No more data available or API limit reached.")
                break

            all_data.extend(data)
            print(f"✅ Batch {i+1}/{batches} complete — total {len(all_data)} rows collected.")
            end_date = start_date
            time.sleep(8)  # ⏳ safe delay for free plan (8 req/min)

        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            break

    if not all_data:
        print("❌ No daily data fetched from TwelveData.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    print(f"🎯 Final merged dataset: {len(df)} DAILY candles of XAU/USD.")
    return df
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
# (rest of your functions: volume profile, fvg, regime, sl/tp, ensemble etc.)
# I'll reuse your previously defined helpers (position_size, calc_sl_tp, train_ensemble, ensemble_predict_proba, fuse_signal, simulate_backtest, build_train_and_signal)
# For brevity, include them unchanged — paste in your version or reuse the earlier implementation.
# (Below I include them verbatim from your provided code so the file is self-contained)
# -----------------------

# Volume profile
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

def regime_detector(df):
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
    sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["Close"].mean()
    trend = df["Close"].iloc[-1] > sma50
    if vol < 0.005 and trend: return "low-vol-trend"
    if vol < 0.005 and not trend: return "low-vol-range"
    if vol >= 0.005: return "high-vol"
    return "unknown"

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
# Build, train, and signal generator (with offline fallback)
# -----------------------
def build_train_and_signal():
    print("📊 Running build_train_and_signal() refresh...")

    # === DAILY DATA FETCH ===
    try:
        daily_df, _ = fetch_fx_daily_xauusd()
        if daily_df.empty:
            print("⚠️ AlphaVantage DAILY failed — trying TwelveData fallback.")
            daily_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1day", total_records=12000)
        if daily_df.empty:
            print("⚠️ Daily TwelveData also failed — will use cached file if available.")
        else:
            daily_df.to_csv("daily.csv", index=False)
            print(f"✅ DAILY data saved locally: {len(daily_df)} rows")
    except Exception as e:
        print("❌ Daily data fetch error:", e)
        daily_df = pd.read_csv("daily.csv") if os.path.exists("daily.csv") else pd.DataFrame()

    # === HOURLY DATA FETCH ===
    try:
        hourly_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1h", total_records=35000)
        if hourly_df.empty:
            print("⚠️ Hourly TwelveData returned no data — will use cached file if available.")
        else:
            hourly_df.to_csv("hourly.csv", index=False)
            print(f"✅ HOURLY data saved locally: {len(hourly_df)} rows")
    except Exception as e:
        print("❌ Hourly data fetch error:", e)
        hourly_df = pd.read_csv("hourly.csv") if os.path.exists("hourly.csv") else pd.DataFrame()

    # === SIGNAL GENERATION PIPELINE ===
    day_out = {"signal": "N/A", "confidence": 0}
    hr_out = {"signal": "N/A", "confidence": 0}

    # === DAILY MODEL PIPELINE ===
    if not daily_df.empty:
        try:
            daily_df.rename(columns=lambda x: x.capitalize(), inplace=True)
            proc = compute_indicators(daily_df)

            # Retrain if model older than 3 days
            if not MODEL_PATH.exists() or (time.time() - MODEL_PATH.stat().st_mtime) > 86400 * 3:
                print("🧠 Retraining ensemble model (daily)...")
                train_ensemble(daily_df)

            model_prob = ensemble_predict_proba(proc)
            fused = fuse_signal(model_prob, proc)
            sltp = calc_sl_tp(proc.iloc[-1], side=fused["signal"] if fused["signal"] in ("BUY", "SELL") else "BUY")
            qty = position_size(sltp["entry"], sltp["sl"])
            day_out = {**fused, **sltp, "qty": qty}
        except Exception as e:
            print("⚠️ Daily pipeline error:", e)

    # === HOURLY MODEL PIPELINE ===
    if not hourly_df.empty:
        try:
            hourly_df.rename(columns=lambda x: x.capitalize(), inplace=True)
            proc_h = compute_indicators(hourly_df)
            model_prob_h = ensemble_predict_proba(proc_h)
            fused_h = fuse_signal(model_prob_h, proc_h)
            sltp_h = calc_sl_tp(proc_h.iloc[-1], side=fused_h["signal"] if fused_h["signal"] in ("BUY", "SELL") else "BUY")
            qty_h = position_size(sltp_h["entry"], sltp_h["sl"])
            hr_out = {**fused_h, **sltp_h, "qty": qty_h}
        except Exception as e:
            print("⚠️ Hourly pipeline error:", e)

    # === OFFLINE FALLBACK (when both APIs failed) ===
    if daily_df.empty and hourly_df.empty:
        print("⚡ Using offline cached data and saved ensemble model for signal regeneration...")
        if os.path.exists("ensemble_model.pkl"):
            with open("ensemble_model.pkl", "rb") as f:
                ensemble = pickle.load(f)
            if os.path.exists("daily.csv") and os.path.exists("hourly.csv"):
                daily_df = pd.read_csv("daily.csv")
                hourly_df = pd.read_csv("hourly.csv")

                proc_d = compute_indicators(daily_df)
                prob_d = ensemble_predict_proba(proc_d)
                fused_d = fuse_signal(prob_d, proc_d)

                proc_h = compute_indicators(hourly_df)
                prob_h = ensemble_predict_proba(proc_h)
                fused_h = fuse_signal(prob_h, proc_h)

                day_out = fused_d
                hr_out = fused_h
                print(f"✅ Offline signal regeneration complete — Daily:{day_out['signal']} | Hourly:{hr_out['signal']}")
            else:
                print("⚠️ No cached CSV data found for offline regeneration.")
        else:
            print("⚠️ No saved ensemble model found — cannot regenerate offline signals.")

    # === SAVE & LOG RESULTS ===
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "daily": day_out,
        "hourly": hr_out
    }

    with open(SIGNALS_FILE, "w") as f:
        json.dump(combined, f, indent=2)

    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.load(open(HISTORY_FILE))
        except Exception:
            history = []
    history.append(combined)
    json.dump(history[-200:], open(HISTORY_FILE, "w"), indent=2)

    print(f"[{combined['timestamp']}] Hourly:{hr_out['signal']} ({hr_out.get('confidence', 0)}%) | "
          f"Daily:{day_out['signal']} ({day_out.get('confidence', 0)}%)")

    return combined

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "active_api_key_tail": ALPHAV_API_KEY[-4:],
        "note": "AlphaVantage primary; TwelveData fallback enabled."
    })

@app.route("/signal")
def signal_route():
    # return saved file if exists; else build on demand
    if SIGNALS_FILE.exists():
        try:
            return jsonify(json.load(open(SIGNALS_FILE)))
        except Exception:
            pass
    # build fresh (this will call fetch_daily/fetch_hourly)
    out = build_train_and_signal()
    return jsonify(out)

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
@app.route("/signal/refresh", methods=["POST"])
# -------------------------
# Auto-refresh loop (every 24h)
# -------------------------
while True:
    try:
        print("🔄 Auto-refreshing signals (daily)...")
        refresh_signals()
        print("✅ Daily signal refresh completed. Waiting 24h...")
    except Exception as e:
        print(f"⚠️ Auto-refresh error: {e}")
    time.sleep(86400)  # wait 24 hours before next refresh
        # Load your saved CSVs
        daily = pd.read_csv("daily.csv")
        hourly = pd.read_csv("hourly.csv")

        # Generate signals
        daily_signal = generate_ensemble_signal(daily, timeframe="daily")
        hourly_signal = generate_ensemble_signal(hourly, timeframe="hourly")

        # Save to signals.json
        import json, datetime
        signal_data = {
            "daily": daily_signal,
            "hourly": hourly_signal,
            "time": datetime.datetime.utcnow().isoformat()
        }
        with open("signals.json", "w") as f:
            json.dump(signal_data, f, indent=2)

        return jsonify({"status": "✅ Signal regenerated", "data": signal_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
     # Force daily-only signal generation
from build import build_train_and_signal, generate_signals_daily_only

# Instead of calling full build_train_and_signal(), call:
generate_signals_daily_only()   
# -----------------------
import pandas as pd
import joblib  # or pickle depending on how you save your model

def generate_signals_daily_only():
    # Load daily CSV
    daily_df = pd.read_csv("daily.csv")  # your saved daily data

    # Load trained daily ensemble
    model = joblib.load("ensemble_daily.pkl")  # adjust filename if different

    # Prepare features (example, adjust to your features)
    X = daily_df[['feature1','feature2','feature3','feature4','feature5','feature6','feature7']]

    # Predict
    daily_df['signal'] = model.predict(X)

    # Save signals
    daily_df.to_csv("daily_signals.csv", index=False)
    print("✅ Daily signals generated and saved to daily_signals.csv")
# Background refresh (run once)
# -----------------------
def background_loop():
    try:
        print("🚀 Running one-time build and signal generation...")
        build_train_and_signal()
        print("✅ Signal generation complete — check /signal endpoint.")
    except Exception as e:
        print("Background loop error:", e)
# -----------------------
# Start server
# -----------------------
if __name__ == "__main__":
    # ✅ Resume from cached data if available
    if DAILY_FILE.exists() and HOURLY_FILE.exists():
        try:
            print("🔄 Cached data found — resuming from saved CSVs...")
            df = pd.read_csv(DAILY_FILE)
            hf = pd.read_csv(HOURLY_FILE)
            print(f"✅ Loaded cached daily ({len(df)}) and hourly ({len(hf)}) rows.")
        except Exception as e:
            print("⚠️ Cache load failed, refetching...", e)
            df, data = fetch_fx_daily_xauusd()
    else:
        print("📥 No cached data found — fetching fresh data...")
        df, data = fetch_fx_daily_xauusd()

    if df.empty:
        print("⚠️ AlphaVantage failed — trying TwelveData fallback.")
        try:
            td_hourly = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1h", total_records=35000)
            td_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1day", total_records=12000)

            if not td_df.empty:
                td_df.rename(columns=lambda x: x.capitalize(), inplace=True)
                print(f"✅ TwelveData DAILY fallback succeeded with {len(td_df)} rows.")
                td_df.to_csv(DAILY_FILE, index=False)
                df = td_df
            else:
                print("❌ TwelveData returned no usable data.")
        except Exception as e:
            print("❌ TwelveData fetch error:", e)

    print(f"🚀 Starting Flask on port {PORT} | Refresh every {REFRESH_INTERVAL} seconds (AlphaVantage + TwelveData enabled)")

    # ✅ Start background loop safely
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=PORT)
# -----------------------
# Start server
# -----------------------
if __name__ == "__main__":
    # ✅ Resume from cached data if available
    if DAILY_FILE.exists() and HOURLY_FILE.exists():
        try:
            print("🔄 Cached data found — resuming from saved CSVs...")
            df = pd.read_csv(DAILY_FILE)
            hf = pd.read_csv(HOURLY_FILE)
            print(f"✅ Loaded cached daily ({len(df)}) and hourly ({len(hf)}) rows.")
        except Exception as e:
            print("⚠️ Cache load failed, refetching...", e)
            df, data = fetch_fx_daily_xauusd()
    else:
        print("📥 No cached data found — fetching fresh data...")
        df, data = fetch_fx_daily_xauusd()

    if df.empty:
        print("⚠️ AlphaVantage failed — trying TwelveData fallback.")
        try:
            td_hourly = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1h", total_records=35000)
            td_df = fetch_twelvedata_xauusd(api_key=TWELVEDATA_KEY, interval="1day", total_records=12000)

            if not td_df.empty:
                td_df.rename(columns=lambda x: x.capitalize(), inplace=True)
                print(f"✅ TwelveData DAILY fallback succeeded with {len(td_df)} rows.")
                td_df.to_csv(DAILY_FILE, index=False)
                df = td_df
            else:
                print("❌ TwelveData returned no usable data.")
        except Exception as e:
            print("❌ TwelveData fetch error:", e)

    print(f"🚀 Starting Flask on port {PORT} | Refresh every {REFRESH_INTERVAL} seconds (AlphaVantage + TwelveData enabled)")

    # ✅ Start background loop safely
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=PORT)
