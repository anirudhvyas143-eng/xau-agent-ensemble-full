# strategy_manager.py

import numpy as np
import pandas as pd
from textblob import TextBlob
import requests
import datetime

# === STRATEGY 1: Trend Following ===
def trend_following(df):
    df["ema_short"] = df["close"].ewm(span=20).mean()
    df["ema_long"] = df["close"].ewm(span=50).mean()
    if df["ema_short"].iloc[-1] > df["ema_long"].iloc[-1]:
        return "BUY", "Trend Up"
    elif df["ema_short"].iloc[-1] < df["ema_long"].iloc[-1]:
        return "SELL", "Trend Down"
    else:
        return "HOLD", "Neutral"

# === STRATEGY 2: Range Trading ===
def range_trading(df):
    recent = df.tail(50)
    high = recent["close"].max()
    low = recent["close"].min()
    current = recent["close"].iloc[-1]
    if current <= low * 1.01:
        return "BUY", "Support Zone"
    elif current >= high * 0.99:
        return "SELL", "Resistance Zone"
    else:
        return "HOLD", "Within Range"

# === STRATEGY 3: Position / Fundamental Trading ===
def position_trading(macro_data):
    # Macro data could be CPI, GDP, interest rates, or gold demand
    if macro_data["inflation"] > 3.0 and macro_data["interest_rate"] < 5.0:
        return "BUY", "Inflation Hedge"
    elif macro_data["interest_rate"] > 5.0:
        return "SELL", "Dollar Strength"
    return "HOLD", "Stable Economy"

# === STRATEGY 4: News / Sentiment Trading ===
def news_sentiment_strategy():
    try:
        # Example free news API (you can later link to NewsAPI or Marketaux)
        url = "https://newsapi.org/v2/everything?q=gold+XAUUSD&apiKey=demo"
        articles = requests.get(url, timeout=10).json().get("articles", [])
        headlines = " ".join(a["title"] for a in articles if "title" in a)
        sentiment = TextBlob(headlines).sentiment.polarity
        if sentiment > 0.2:
            return "BUY", f"Positive Sentiment ({sentiment:.2f})"
        elif sentiment < -0.2:
            return "SELL", f"Negative Sentiment ({sentiment:.2f})"
        else:
            return "HOLD", "Neutral Sentiment"
    except Exception as e:
        return "HOLD", f"News Error: {e}"

# === FINAL STRATEGY AGGREGATOR ===
def combined_signal(df, macro_data):
    trend_sig, trend_reason = trend_following(df)
    range_sig, range_reason = range_trading(df)
    pos_sig, pos_reason = position_trading(macro_data)
    news_sig, news_reason = news_sentiment_strategy()

    signals = [trend_sig, range_sig, pos_sig, news_sig]
    signal_strength = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL")}

    if signal_strength["BUY"] > signal_strength["SELL"]:
        final = "BUY"
    elif signal_strength["SELL"] > signal_strength["BUY"]:
        final = "SELL"
    else:
        final = "HOLD"

    summary = {
        "trend": trend_reason,
        "range": range_reason,
        "fundamental": pos_reason,
        "news": news_reason
    }

    return final, summary
