import os
import pandas as pd
from main import build_train_and_signal  # adjust if your main.py function name is different

# --- Step 1: Save daily and hourly data ---
try:
    # If already in memory
    daily_data
except NameError:
    print("daily_data not found in memory. Loading/fetching daily data...")
    # Replace this with your existing fetching code
    # Example: daily_data = fetch_daily_data_from_api()
    daily_data = pd.DataFrame()  # temporary placeholder

try:
    hourly_data
except NameError:
    print("hourly_data not found in memory. Loading/fetching hourly data...")
    # Replace this with your existing fetching code
    # Example: hourly_data = fetch_hourly_data_from_api()
    hourly_data = pd.DataFrame()  # temporary placeholder

# --- Step 2: Save CSVs ---
daily_csv_path = os.path.join(os.getcwd(), 'daily.csv')
hourly_csv_path = os.path.join(os.getcwd(), 'hourly.csv')

if not daily_data.empty:
    daily_data.to_csv(daily_csv_path, index=False)
    print(f"Saved daily data: {daily_csv_path}")
else:
    print("Warning: daily_data is empty. Server may still fail without proper daily rows.")

if not hourly_data.empty:
    hourly_data.to_csv(hourly_csv_path, index=False)
    print(f"Saved hourly data: {hourly_csv_path}")
else:
    print("Warning: hourly_data is empty. Server may still fail without proper hourly rows.")

# --- Step 3: Retrain model ---
try:
    build_train_and_signal()  # or the function your main.py uses to train and refresh
    print("Model retraining and signal generation complete.")
except Exception as e:
    print("Error during model training:", e)

# --- Step 4: Server ready ---
print("Server is now ready. Your bookmarklet should work after this.")
