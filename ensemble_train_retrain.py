# ensemble_train_retrain.py
import os, joblib
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(".").resolve()
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

def train_simple_rf(X, y, name="rf_model"):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(Xs, y)
    joblib.dump(model, MODELS / f"{name}.pkl")
    joblib.dump(scaler, MODELS / f"{name}_scaler.pkl")
    return model
