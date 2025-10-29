"""
AI Automation Engine for XAUUSD Agent
-------------------------------------
Runs hourly + daily retraining, backtesting, and signal logging.
Integrates Quant Intelligence, Feature Engineering, and Model Drift Correction.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from ensemble_train_retrain import model as train_model
from ensemble_inferno_and_data import result as infer_signal
from backtest_engine import compute_metrics
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_FILE = "features_full_daily.csv"
MODEL_PATH = "model.pkl"
HISTORY_FILE = "signal_log.csv"
BACKTEST_DIR = "backtests"
os.makedirs(BACKTEST_DIR, exist_ok=True)

# --------------------------------------------
# 🔁 Utility: Hyperparameter Optimization
# --------------------------------------------
def optimize_model(data):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        X = data[['ema21', 'ema50', 'atr14']].fillna(method='bfill')
        y = (data['price'].shift(-1) > data['price']).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    return study.best_params

# --------------------------------------------
# 🧠 AI Reinforcement Update
# --------------------------------------------
def reinforcement_correction(prob, prev_outcome):
    # Basic reward adjustment logic
    if prev_outcome == 1:
        return min(1.0, prob + 0.05)
    else:
        return max(0.0, prob - 0.05)

# --------------------------------------------
# 🚀 Main Routine
# --------------------------------------------
def run_ai_cycle():
    print("\n🧠 Starting AI Retrain + Inference Cycle...")
    
    data = pd.read_csv(DATA_FILE)
    best_params = optimize_model(data)
    print(f"✅ Optimized Params: {best_params}")

    # Retrain model
    model = RandomForestClassifier(**best_params, random_state=42)
    X = data[['ema21', 'ema50', 'atr14']].fillna(method='bfill')
    y = (data['price'].shift(-1) > data['price']).astype(int)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model retrained successfully.")

    # Run inference
    last_row = X.iloc[-1].values.reshape(1, -1)
    pred = model.predict(last_row)[0]
    prob = model.predict_proba(last_row)[0][pred]
    prob_adj = reinforcement_correction(prob, pred)
    
    signal = "BUY" if pred == 1 else "SELL"
    timestamp = datetime.utcnow().isoformat()

    # Log to history
    log_df = pd.DataFrame([[timestamp, signal, round(prob_adj, 4)]],
                          columns=["timestamp", "signal", "confidence"])
    log_df.to_csv(HISTORY_FILE, mode='a', index=False, header=not os.path.exists(HISTORY_FILE))
    print(f"🪙 Signal: {signal} ({prob_adj:.2%})")

    # Backtest update
    from backtest_engine import metrics as bt_metrics
    print("📊 Backtest metrics:")
    for k, v in bt_metrics.items():
        print(f"   {k:15s}: {v:.2%}")

    print(f"Cycle complete ✅\n")

if __name__ == "__main__":
    while True:
        run_ai_cycle()
        # Run hourly (3600s), daily (24h) cycles can be handled by Render cron
        time.sleep(3600)
