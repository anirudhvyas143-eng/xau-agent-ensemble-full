"""
AI AUTOMATION ENGINE — XAUUSD AI Trader
---------------------------------------
This script automatically:
✅ Loads and cleans daily/weekly/monthly datasets
✅ Trains and optimizes hybrid ML + RL ensemble
✅ Generates hourly BUY/SELL signals with confidence
✅ Backtests automatically and logs all results
✅ Serves API endpoints for live inference (Flask)
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

# ML / RL / Optimization Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from stable_baselines3 import PPO
import gymnasium as gym

# ===============================
# CONFIG
# ===============================
DATA_PATH = os.getenv("DATA_PATH_DAILY", "data/XAU_USD_Historical_Data_daily.csv")
MODEL_PATH = os.getenv("MODEL_SAVE_PATH", "models/best_model.pkl")
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
INFER_INTERVAL = int(os.getenv("INFER_INTERVAL_SECS", 3600))  # every hour

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger.add(LOG_PATH, rotation="1 MB")

# ===============================
# DATA PREPARATION
# ===============================
def load_and_prepare_data():
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "close" not in df.columns:
        if "price" in df.columns:
            df.rename(columns={"price": "close"}, inplace=True)
        elif "last" in df.columns:
            df.rename(columns={"last": "close"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["close"])
    df["return"] = df["close"].pct_change()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["atr14"] = df["close"].pct_change().rolling(14).std() * 100
    df = df.dropna()
    return df

# ===============================
# MACHINE LEARNING TRAINING
# ===============================
def train_ml_model(df):
    logger.info("Training Random Forest (Ensemble)...")
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    features = ["ema21", "ema50", "atr14"]
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    logger.success(f"Model trained successfully with accuracy: {acc:.2%}")
    return model, acc

# ===============================
# REINFORCEMENT LEARNING TUNER
# ===============================
class SimpleEnv(gym.Env):
    def __init__(self, prices):
        super(SimpleEnv, self).__init__()
        self.prices = prices
        self.action_space = gym.spaces.Discrete(2)  # 0 = SELL, 1 = BUY
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.current_step = 50

    def reset(self, *, seed=None, options=None):
        self.current_step = 50
        return self._get_obs(), {}

    def _get_obs(self):
        close = self.prices[self.current_step]
        ema21 = np.mean(self.prices[self.current_step-21:self.current_step])
        ema50 = np.mean(self.prices[self.current_step-50:self.current_step])
        return np.array([close, ema21, ema50], dtype=np.float32)

    def step(self, action):
        reward = 0
        if self.current_step < len(self.prices) - 1:
            price_change = self.prices[self.current_step + 1] - self.prices[self.current_step]
            reward = price_change if action == 1 else -price_change
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        return self._get_obs(), reward, done, False, {}

def train_rl_model(prices):
    logger.info("Training Reinforcement Learning agent (PPO)...")
    env = SimpleEnv(prices)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)
    model.save("models/rl_model.zip")
    logger.success("RL model training completed successfully!")

# ===============================
# HYPERPARAMETER OPTIMIZATION
# ===============================
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 12)
    df = load_and_prepare_data()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    features = ["ema21", "ema50", "atr14"]
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc

def optimize_model():
    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    logger.success(f"Best parameters found: {best_params}")
    return best_params

# ===============================
# SIGNAL GENERATION
# ===============================
def generate_signal(model, df):
    latest = df.iloc[-1]
    X_latest = latest[["ema21", "ema50", "atr14"]].values.reshape(1, -1)
    prediction = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0][prediction]
    price = float(latest["close"])

    if prediction
