import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================================
# 🧠 Ensemble Model Trainer
# ================================
# Purpose:
# - Trains or retrains a Random Forest model
# - Uses processed features from XAUUSD historical data
# - Auto-saves to /models for Render / local integration
# ================================

# --- Configuration ---
DATA_PATH = os.getenv("FEATURE_FILE", "features_full_daily.csv")
MODEL_PATH = os.getenv("MODEL_SAVE_PATH", "models/rf_model.joblib")

# --- Load data ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Feature file not found: {DATA_PATH}")

print(f"📥 Loading feature data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# --- Target definition ---
# Predict whether price will rise the next day
data['target'] = (data['price'].shift(-1) > data['price']).astype(int)

# --- Feature selection ---
# Expand this list as your feature engineering grows
features = [col for col in data.columns if col.startswith(("ema", "atr", "rsi", "volatility", "momentum"))]
if not features:
    features = ['ema21', 'ema50', 'atr14']

X = data[features].fillna(method='bfill')
y = data['target']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Train Random Forest ---
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- Save model ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")

# --- Evaluate performance ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2%}")

# --- Optional retrain trigger log ---
with open("train_log.txt", "a") as log:
    log.write(f"Retrain completed — Accuracy: {acc:.4f}\n")
