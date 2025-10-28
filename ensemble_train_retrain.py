import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('features_full_daily.csv')
data['target'] = (data['price'].shift(-1) > data['price']).astype(int)
features = ['ema21', 'ema50', 'atr14']
X = data[features].fillna(method='bfill')
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained. Accuracy: {acc:.2%}")
