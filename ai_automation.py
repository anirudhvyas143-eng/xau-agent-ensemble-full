# ai_automation.py
"""
AI Automation orchestrator for XAUUSD project.
- Loads data (via load_data.load_and_prepare)
- Trains / retrains optimized RandomForest (Optuna)
- Optional RL fine-tune (PPO)
- Generates latest BUY/SELL signal and saves latest_signal.json
- Runs scheduled retrain + inference
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

# Optional RL (kept safe/guarded)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.envs import DummyVecEnv
    import gymnasium as gym
    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False

# Project modules (make sure these files exist in repo)
from load_data import load_and_prepare         # returns dict of dataframes keyed by timeframe
from strategy_manager import select_strategy  # your dynamic strategy selector
from drift_manager import detect_model_drift  # drift detection helper

# ---------------------------
# Configuration (env override)
# ---------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/best_model.pkl")
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
INFER_INTERVAL = int(os.getenv("INFER_INTERVAL_SECS", 3600))  # default 1 hour
RETRAIN_HOURS = int(os.getenv("RETRAIN_HOURS", 24))
PORT = int(os.getenv("PORT", 10000))

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger.add(LOG_PATH, rotation="1 day", level="INFO")
logger.info("🚀 AI Automation System Initialized")

# ---------------------------
# Helper: prepare training data
# ---------------------------
def prepare_training_df():
    """
    Use load_and_prepare() from load_data.py to fetch and compute features.
    Returns a cleaned daily dataframe suitable for supervised training.
    """
    datasets = load_and_prepare()
    if not isinstance(datasets, dict):
        raise RuntimeError("load_and_prepare() must return a dict of dataframes keyed by timeframe")

    df_daily = datasets.get("daily")
    if df_daily is None or df_daily.empty:
        raise RuntimeError("No daily data available for training")

    # Ensure 'close' column present (load_and_prepare should have normalized)
    if "close" not in df_daily.columns:
        # try common alternatives
        for alt in ["Close", "price", "last"]:
            if alt in df_daily.columns:
                df_daily["close"] = df_daily[alt]
                break
    df = df_daily.copy().sort_index()
    # basic features expected by the pipeline
    if "ema_20" in df.columns:
        df.rename(columns={"ema_20": "ema20", "ema_50": "ema50"}, inplace=True, errors="ignore")
    # create simple training features if not present
    if "ema21" not in df.columns:
        df["ema21"] = df["close"].ewm(span=21).mean()
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50).mean()
    if "atr_14" not in df.columns and "atr_14" not in df.columns:
        df["return"] = df["close"].pct_change()
        df["atr14"] = df["return"].rolling(14).std().fillna(0)
    else:
        if "atr_14" in df.columns:
            df["atr14"] = df["atr_14"]
        elif "atr_14" not in df.columns and "atr14" not in df.columns:
            df["atr14"] = df["return"].rolling(14).std().fillna(0)

    # keep necessary columns
    df = df.reset_index()
    if "date" not in df.columns and df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "date"})
    df = df.dropna(subset=["close", "ema21", "ema50", "atr14"])
    return df

# ---------------------------
# Model training (Optuna)
# ---------------------------
def train_optuna_model(df, n_trials=15):
    """Train RandomForest optimized by Optuna on the prepared df."""
    logger.info("🔍 Starting Optuna tuning for RandomForest...")
    # create label (next-day up)
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["ema21", "ema50", "atr14"]].fillna(0)
    y = df["target"].fillna(0).astype(int)

    # require enough rows
    if len(X) < 40:
        logger.warning("Not enough rows to run Optuna tuning; training baseline RF without tuning.")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        joblib.dump(rf, MODEL_SAVE_PATH)
        logger.info("💾 Baseline model saved (no tuning).")
        return rf

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info(f"🏆 Optuna best params: {best}")

    final = RandomForestClassifier(**best, random_state=42, n_jobs=-1)
    final.fit(X, y)
    joblib.dump(final, MODEL_SAVE_PATH)
    logger.info(f"💾 Final model trained and saved to {MODEL_SAVE_PATH}")
    return final

# ---------------------------
# Optional RL trainer (safe/guarded)
# ---------------------------
def reinforcement_training(df, total_timesteps=8000):
    """Optional RL tuning — wrapped in try/except and runs only if RL libs available."""
    if not RL_AVAILABLE:
        logger.warning("Stable-baselines3 / gym not available — skipping RL training.")
        return None
    try:
        logger.info("🤖 Starting optional PPO training (this may take a while)...")

        class TradingEnv(gym.Env):
            def __init__(self, data):
                super().__init__()
                self.data = data.reset_index(drop=True)
                self.current_step = 0
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(3)

            def reset(self, seed=None, options=None):
                self.current_step = 0
                return self._get_obs(), {}

            def _get_obs(self):
                row = self.data.iloc[self.current_step]
                return np.array([row["ema21"], row["ema50"], row["atr14"]], dtype=np.float32)

            def step(self, action):
                reward = 0.0
                # next index reward proxy
                if self.current_step + 1 < len(self.data):
                    reward = float(self.data.iloc[self.current_step + 1]["close"] / (self.data.iloc[self.current_step]["close"]) - 1.0)
                self.current_step += 1
                done = self.current_step >= len(self.data) - 2
                return self._get_obs(), reward, done, False, {}

        env = DummyVecEnv([lambda: TradingEnv(df)])
        agent = PPO("MlpPolicy", env, verbose=0)
        agent.learn(total_timesteps=total_timesteps)
        agent.save("models/ppo_agent.zip")
        logger.info("✅ PPO agent trained and saved (models/ppo_agent.zip)")
        return agent
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        return None

# ---------------------------
# Signal generation
# ---------------------------
def generate_signal(df, model):
    """Generates latest BUY/SELL signal using adaptive strategy selection."""
    if df is None or df.empty:
        raise RuntimeError("Empty input dataframe for signal generation")
    last = df.reset_index().iloc[-1]
    X_latest = last[["ema21", "ema50", "atr14"]].values.reshape(1, -1)

    # safe model load/check
    if model is None:
        raise RuntimeError("Model is None when generating signal")

    prediction = int(model.predict(X_latest)[0])
    probabilities = model.predict_proba(X_latest)[0] if hasattr(model, "predict_proba") else [0.0, 0.0]
    confidence = float(np.max(probabilities))
    price = float(last["close"])

    strategy = "default"
    try:
        strategy = select_strategy(df)  # returns e.g. "trend_following", "range_trading", etc.
    except Exception as e:
        logger.warning(f"Strategy selection failed, using default. Err: {e}")

    # Adjust targets using chosen strategy
    if prediction == 1:
        signal = "BUY"
        if strategy == "trend_following":
            tp, sl = price * 1.010, price * 0.993
        elif strategy == "range_trading":
            tp, sl = price * 1.004, price * 0.996
        elif strategy == "news_trading":
            tp, sl = price * 1.015, price * 0.990
        else:
            tp, sl = price * 1.020, price * 0.985
        entry = price * 0.999
    else:
        signal = "SELL"
        if strategy == "trend_following":
            tp, sl = price * 0.990, price * 1.007
        elif strategy == "range_trading":
            tp, sl = price * 0.996, price * 1.004
        elif strategy == "news_trading":
            tp, sl = price * 0.985, price * 1.010
        else:
            tp, sl = price * 0.980, price * 1.015
        entry = price * 1.001

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy,
        "signal": signal,
        "confidence": round(confidence, 4),
        "entry_price": round(entry, 4),
        "take_profit": round(tp, 4),
        "stop_loss": round(sl, 4),
        "current_price": round(price, 4),
    }
    logger.info(f"📈 Signal Generated ({strategy}): {result}")
    return result

# ---------------------------
# Background jobs: retrain + inference
# ---------------------------
def retrain_job():
    """Scheduled retrain job (daily/periodic)."""
    logger.info("♻️ Retrain job started")
    try:
        df = prepare_training_df()
        model = train_optuna_model(df, n_trials=15)
        # optional RL fine-tune (guarded)
        try:
            reinforcement_training(df)
        except Exception as e:
            logger.warning(f"Reinforcement fine-tune skipped/failed: {e}")
        logger.info("♻️ Retrain job completed")
        return model
    except Exception as e:
        logger.error(f"Retrain job failed: {e}")
        return None

def inference_job():
    """Scheduled inference job (hourly by default)."""
    logger.info("💡 Inference job started")
    try:
        df = prepare_training_df()
    except Exception as e:
        logger.error(f"Inference aborted — data prepare failed: {e}")
        return

    # load or retrain if model missing
    model = None
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model = joblib.load(MODEL_SAVE_PATH)
        except Exception as e:
            logger.error(f"Failed to load model at {MODEL_SAVE_PATH}: {e}")
            model = None

    if model is None:
        logger.warning("Model missing or failed to load — running immediate retrain.")
        model = retrain_job()
        if model is None and os.path.exists(MODEL_SAVE_PATH):
            try:
                model = joblib.load(MODEL_SAVE_PATH)
            except Exception as e:
                logger.error(f"Still cannot load model after retrain: {e}")
                return

    # Generate signal
    try:
        signal = generate_signal(df, model)
        with open("latest_signal.json", "w") as f:
            json.dump(signal, f, indent=2)
        logger.info("💡 Latest signal saved to latest_signal.json")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return

    # Run drift detection (non-blocking)
    try:
        detect_model_drift(df, MODEL_SAVE_PATH)
    except Exception as e:
        logger.warning(f"Drift detection failed: {e}")

# ---------------------------
# Flask API
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/signal", methods=["GET"])
def get_signal():
    if not os.path.exists("latest_signal.json"):
        inference_job()
    try:
        with open("latest_signal.json") as f:
            signal = json.load(f)
        return jsonify(signal)
    except Exception as e:
        logger.error(f"Failed returning latest signal: {e}")
        return jsonify({"error": "no signal available", "message": str(e)}), 500

# ---------------------------
# Scheduler start
# ---------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_job, "interval", hours=RETRAIN_HOURS)
scheduler.add_job(inference_job, "interval", seconds=INFER_INTERVAL)
scheduler.start()
logger.info("🕒 Scheduler started (auto retrain + inference)")

if __name__ == "__main__":
    # Run once on startup synchronously (safe)
    try:
        retrain_job()
    except Exception as e:
        logger.warning(f"Startup retrain failed: {e}")
    try:
        inference_job()
    except Exception as e:
        logger.warning(f"Startup inference failed: {e}")

    logger.info(f"🌍 Starting Flask API on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
