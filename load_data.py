import pandas as pd
import numpy as np
import os

def load_and_merge():
    paths = [
        'data/XAU_USD_Historical_Data_daily.csv',
        'data/XAU_USD_Historical_Data_weekly.csv',
        'data/XAU_USD_Historical_Data_monthly.csv'
    ]

    frames = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            frames.append(df)

    data = pd.concat(frames).drop_duplicates('date').sort_values('date')
    data['ema21'] = data['price'].ewm(span=21).mean()
    data['ema50'] = data['price'].ewm(span=50).mean()
    data['atr14'] = (data['high'] - data['low']).rolling(14).mean()
    data.to_csv('features_full_daily.csv', index=False)
    print(f"✅ Saved processed data: {len(data)} rows")

if __name__ == "__main__":
    load_and_merge()
