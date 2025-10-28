import pandas as pd, joblib, datetime, json

data = pd.read_csv('features_full_daily.csv')
model = joblib.load('model.pkl')

last = data.iloc[-1]
