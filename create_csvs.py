# create_csvs.py
import pandas as pd

# Replace these with your actual fetched data if available
daily_df = pd.DataFrame(...)   # your daily data here
hourly_df = pd.DataFrame(...)  # your hourly data here

# Save CSVs
daily_df.to_csv("daily.csv", index=False)
hourly_df.to_csv("hourly.csv", index=False)

print("✅ CSVs created: daily.csv & hourly.csv")
