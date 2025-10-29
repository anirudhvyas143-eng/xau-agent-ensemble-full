import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gymnasium as gym
from drift_manager import detect_model_drift
from strategy_manager import select_strategy

# ===============================
# CONFIGURATION
# ===============================
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/best_model.pkl")
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
INFER_INTERVAL = int(os.getenv("INFER_INTERVAL_SECS", 3600))  # every hour
PORT = int(os.getenv("PORT", 10000))

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger.add(LOG_PATH, rotation="1 day", level="INFO")
logger.info("🚀 AI Automation System Initialized")

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_data():
    """Loads and prepares historical data."""
    path = os.path.join(DATA_DIR, "XAU_USD_Historical_Data_daily.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "price" in df.columns:
        df.rename(columns={"price": "close"}, inplace=True)
    elif "close" not in df.columns and "last" in df.columns:
        df.rename(columns={"last": "close"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["close"])
    df["return"] = df["close"].pct_change()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr14"] = df["return"].rolling(14).std()
    df.dropna(inplace=True)

    logger.info(f"✅ Data loaded: {len(df)} rows")
    return df


def train_optuna_model(df):
    """Trains and optimizes RandomForest with Optuna."""
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["ema21", "ema50", "atr14"]]
    y = df["target"]

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("🔍 Starting Optuna tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best_params = study.best_params
    logger.info(f"🏆 Best Params: {best_params}")

    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    logger.info(f"💾 Model saved to {MODEL_SAVE_PATH}")
    return final_model


def reinforcement_training(df):
    """Optional: train PPO agent to refine long/short timing."""
    logger.info("🤖 Starting PPO Reinforcement Learning fine-tune...")

    class TradingEnv(gym.Env):
        def __init__(self, data):
            super().__init__()
            self.data = data.reset_index(drop=True)
            self.current_step = 0
            self.balance = 10000
            self.position = 0
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = 10000
            self.position = 0
            return self._get_obs(), {}

        def _get_obs(self):
            row = self.data.iloc[self.current_step]
            return np.array([row["ema21"], row["ema50"], row["atr14"]], dtype=np.float32)

        def step(self, action):
            price = self.data.iloc[self.current_step]["close"]
            reward = 0

            if action == 1:  # BUY
                self.position += 1
                reward = self.data.iloc[self.current_step + 1]["return"]
            elif action == 2:  # SELL
                self.position -= 1
                reward = -self.data.iloc[self.current_step + 1]["return"]

            self.balance *= (1 + reward)
            self.current_step += 1
            done = self.current_step >= len(self.data) - 2
            return self._get_obs(), reward, done, False, {}

    env = DummyVecEnv([lambda: TradingEnv(df)])
    agent = PPO("MlpPolicy", env, verbose=0)
    agent.learn(total_timesteps=10000)
    agent.save("models/ppo_agent.zip")
    logger.info("✅ PPO agent saved (models/ppo_agent.zip)")

def generate_signal(df, model):
    """Generates latest BUY/SELL signal using adaptive strategy."""
    last = df.iloc[-1]
    X_latest = last[["ema21", "ema50", "atr14"]].values.reshape(1, -1)
    prediction = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0][prediction]
    price = float(last["close"])

    # --- Select best-fit strategy dynamically ---
    strategy = select_strategy(df)
    logger.info(f"🎯 Selected Strategy: {strategy}")

    # --- Adjust logic based on strategy ---
    if prediction == 1:
        signal = "BUY"
        if strategy == "trend_following":
            tp = price * 1.010
            sl = price * 0.993
        elif strategy == "range_trading":
            tp = price * 1.004
            sl = price * 0.996
        elif strategy == "news_trading":
            tp = price * 1.015
            sl = price * 0.990
        else:  # position_trading
            tp = price * 1.020
            sl = price * 0.985
        entry = price * 0.999
    else:
        signal = "SELL"
        if strategy == "trend_following":
            tp = price * 0.990
            sl = price * 1.007
        elif strategy == "range_trading":
            tp = price * 0.996
            sl = price * 1.004
        elif strategy == "news_trading":
            tp = price * 0.985
            sl = price * 1.010
        else:  # position_trading
            tp = price * 0.980
            sl = price * 1.015
        entry = price * 1.001

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy,
        "signal": signal,
        "confidence": round(float(prob), 4),
        "entry_price": round(entry, 2),
        "take_profit": round(tp, 2),
        "stop_loss": round(sl, 2),
        "current_price": round(price, 2),
    }

    logger.info(f"📈 Signal Generated ({strategy}): {result}")
    return result



# ===============================
# BACKGROUND JOBS
# ===============================
def retrain_job():
    df = load_data()
    model = train_optuna_model(df)
    reinforcement_training(df)
    logger.info("♻️ Retraining complete")


def inference_job():
    df = load_data()
    model = joblib.load(MODEL_SAVE_PATH)
    signal = generate_signal(df, model)
    with open("latest_signal.json", "w") as f:
        json.dump(signal, f, indent=2)
    logger.info("💡 Latest signal saved to latest_signal.json")
        detect_model_drift(df)


# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)
CORS(app)

@app.route("/signal", methods=["GET"])
def get_signal():
    if not os.path.exists("latest_signal.json"):
        inference_job()
    with open("latest_signal.json") as f:
        signal = json.load(f)
    return jsonify(signal)

# ===============================
# SCHEDULER + STARTUP
# ===============================
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_job, "interval", hours=24)
scheduler.add_job(inference_job, "interval", seconds=INFER_INTERVAL)
scheduler.start()
logger.info("🕒 Scheduler started (auto retrain + inference)")

if __name__ == "__main__":
    retrain_job()
    inference_job()
    logger.info(f"🌍 Starting Flask API on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
