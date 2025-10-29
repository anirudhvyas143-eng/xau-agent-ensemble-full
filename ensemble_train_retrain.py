import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger
import optuna

# =========================================
# ⚙️ CONFIG
# =========================================
DATA_DIR = "data"
MODEL_PATH = "models/ensemble_model.pkl"
LOG_PATH = "logs/ensemble_train.log"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add(LOG_PATH, rotation="1 day", level="INFO")

# =========================================
# 🧩 Load & Merge Data (multi-timeframe)
# =========================================
def load_all_features():
    files = [
        "features_full_daily.csv",
        "features_full_weekly.csv",
        "features_full_monthly.csv"
    ]
    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timeframe"] = f.split("_")[-1].replace(".csv", "")
            dfs.append(df)
            logger.info(f"✅ Loaded {f} ({len(df)} rows)")
        else:
            logger.warning(f"⚠️ Missing: {f}")
    if not dfs:
        raise FileNotFoundError("❌ No feature files found in /data/")
    df = pd.concat(dfs, axis=0).dropna(subset=["Close"], how="any")
    return df


# =========================================
# 🔍 Feature Prep + Target
# =========================================
def prepare_data(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c not in ["target", "Date", "timeframe", "Close"]]
    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


# =========================================
# 🎯 OPTUNA Optimization (RandomForest)
# =========================================
def optimize_rf(X, y):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 400)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("🔍 Running Optuna optimization for RandomForest...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    return study.best_params


# =========================================
# 🧠 Train Ensemble Model
# =========================================
def train_ensemble():
    df = load_all_features()
    X, y, features = prepare_data(df)
    logger.info(f"📊 Training on {len(X)} samples, {len(features)} features")

    best_rf_params = optimize_rf(X, y)
    logger.info(f"🏆 Best RF Params: {best_rf_params}")

    rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[0.6, 0.4]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": ensemble, "features": features}, MODEL_PATH)

    logger.success(f"✅ Ensemble model trained and saved → {MODEL_PATH}")
    logger.info(f"📈 Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))
    return acc


if __name__ == "__main__":
    logger.info("🚀 Starting Ensemble Model Training ...")
    accuracy = train_ensemble()
    logger.info(f"🏁 Final Accuracy: {accuracy:.4f}")
