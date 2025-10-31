# load_data.py — modular data fetchers (Investing + AlphaVantage)
import requests, pandas as pd, os
from pathlib import Path
from datetime import datetime

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DAILY_FILE = DATA_DIR / "XAU_USD_Historical_Data_daily.csv"
HOURLY_FILE = DATA_DIR / "XAU_USD_Historical_Data_hourly.csv"

def fetch_investing_daily(api_key, save_path=DAILY_FILE):
    """Thin wrapper to call Investing RapidAPI (same logic as in main)."""
    url = "https://investing-com.p.rapidapi.com/price/historical"
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "investing-com.p.rapidapi.com"}
    params = {"symbol": "XAU/USD", "interval": "1d", "from": "2000-01-01"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    data = r.json()
    df = pd.DataFrame(data["data"])
    df["Date"] = pd.to_datetime(df["date"])
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume":"Volume"}, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])].sort_values("Date")
    df.to_csv(save_path, index=False)
    return df

def fetch_alpha_hourly(api_key, save_path=HOURLY_FILE):
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY"
        f"&from_symbol=XAU&to_symbol=USD&interval=60min"
        f"&apikey={api_key}&outputsize=full"
    )
    r = requests.get(url, timeout=15)
    data = r.json().get("Time Series FX (60min)", {})
    df = pd.DataFrame(data).T
    df.columns = ["Open", "High", "Low", "Close"]
    df = df.astype(float)
    df["Date"] = pd.to_datetime(df.index)
    df = df.sort_values("Date")
    df.to_csv(save_path, index=False)
    return df
