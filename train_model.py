import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import optuna
from loguru import logger

# === CONFIG ===
DATA_PATH = "data/features_ensemble.csv"
MODEL_PATH = "models/best_model.pkl"

os.makedirs("models", exist_ok=True)
logger.add("logs/train.log", rotation="1 day", level="INFO")

# === HELPER: Load Data ===
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Ensemble data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0)
    logger.info(f"📊 Loaded ensemble dataset: {df.shape} rows")
    
    # Ensure correct columns
    df = df.dropna(subset=["close"])
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c not in ["target", "timeframe", "date", "close"]]
    X = df[feature_cols]
    y = df["target"]

    return X, y, feature_cols


# === OPTUNA Optimization ===
def optimize_model(X, y):
    logger.info("🎯 Starting Optuna optimization...")

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params
    logger.info(f"🏆 Best Params Found: {best_params}")
    return best_params


# === Train Final Model ===
def train_final_model():
    X, y, feature_cols = load_data()
    best_params = optimize_model(X, y)

    logger.info("🧠 Training final RandomForest model...")
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    logger.success(f"💾 Model saved → {MODEL_PATH}")

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    logger.info(f"✅ Training complete. Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y, preds))


if __name__ == "__main__":
    logger.info("🚀 Starting model training pipeline...")
    train_final_model()
