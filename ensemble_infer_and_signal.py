# ensemble_infer_and_signal.py
import joblib, numpy as np
from pathlib import Path

ROOT = Path(".").resolve()
MODELS = ROOT / "models"

def infer_model(model_name, X):
    try:
        model = joblib.load(MODELS / f"{model_name}.pkl")
        scaler = joblib.load(MODELS / f"{model_name}_scaler.pkl")
    except Exception:
        return None
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0][1]
    pred = int(model.predict(Xs)[0])
    return {"pred": pred, "prob": float(prob)}
