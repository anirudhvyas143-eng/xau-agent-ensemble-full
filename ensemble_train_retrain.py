import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('features_full_daily.csv')

# Define target (1 = price will go up next day, 0 = price will fall)
data['target'] = (data['price'].shift(-1) > data['price']).astype(int)

# Select important features
features = ['ema21', 'ema50', 'atr14']
X = data[features].fillna(method='bfill')
y = data['target']

# Split into train-test sets (no shuffle to preserve time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'model.pkl')

# Evaluate performance
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained successfully! Accuracy: {acc:.2%}")
