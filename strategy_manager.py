# strategy_manager.py

import numpy as np
import pandas as pd
from textblob import TextBlob
import requests
import datetime
import logging

# === STRATEGY 1: Trend Following ===
def trend_following(df):
    """EMA-based trend following strategy."""
    if len(df) < 50:
        return "HOLD", "Insufficient data"

    if "ema_short" not in df.columns:
        df["ema_short"] = df["close"].ewm(span=20).mean()
    if "ema_long" not in df.columns:
        df["ema_long"] = df["close"].ewm(span=50).mean()

    if df["ema_short"].iloc[-1] > df["ema_long"].iloc[-1]:
        return "BUY", "Trend Up"
    elif df["ema_short"].iloc[-1] < df["ema_long"].iloc[-1]:
        return "SELL", "Trend Down"
    else:
        return "HOLD", "Neutral"


# === STRATEGY 2: Range Trading ===
def range_trading(df):
    """Identify support and resistance zones for range trading."""
    if len(df) < 50:
        return "HOLD", "Not enough data for range detection"

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
    """Macro-based strategy using inflation and interest rates."""
    inflation = macro_data.get("inflation", 2.0)
    interest_rate = macro_data.get("interest_rate", 4.0)

    if inflation > 3.0 and interest_rate < 5.0:
        return "BUY", "Inflation Hedge"
    elif interest_rate > 5.0:
        return "SELL", "Dollar Strength"
    else:
        return "HOLD", "Stable Economy"


# === STRATEGY 4: News / Sentiment Trading ===
def news_sentiment_strategy():
    """Analyze market sentiment from news headlines."""
    try:
        url = "https://newsapi.org/v2/everything?q=gold+XAUUSD&apiKey=demo"
        response = requests.get(url, timeout=10)
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return "HOLD", "No recent news"

        headlines = " ".join(a.get("title", "") for a in articles)
        sentiment = TextBlob(headlines).sentiment.polarity

        if sentiment > 0.2:
            return "BUY", f"Positive Sentiment ({sentiment:.2f})"
        elif sentiment < -0.2:
            return "SELL", f"Negative Sentiment ({sentiment:.2f})"
        else:
            return "HOLD", "Neutral Sentiment"
    except Exception as e:
        return "HOLD", f"News Error: {str(e)}"


# === STRATEGY 5: Combined Aggregator ===
def combined_signal(df, macro_data):
    """Combine multiple strategies into one unified signal."""
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


# === STRATEGY SELECTION ENGINE ===
def select_strategy(df):
    """
    Dynamically select the dominant strategy based on market structure.
    Returns:
      - 'trend_following'
      - 'range_trading'
      - 'news_trading'
      - 'position_trading'
      - 'default'
    """
    try:
        if len(df) < 50:
            return "default"

        # Compute EMAs and volatility
        if "ema20" not in df.columns:
            df["ema20"] = df["close"].ewm(span=20).mean()
        if "ema50" not in df.columns:
            df["ema50"] = df["close"].ewm(span=50).mean()

        ema_diff = abs(df["ema20"].iloc[-1] - df["ema50"].iloc[-1])
        price_std = df["close"].tail(50).std()
        avg_price = df["close"].tail(50).mean()
        volatility = (price_std / avg_price) * 100

        # Logic for strategy selection
        if ema_diff / df["close"].iloc[-1] > 0.005 and volatility > 1.0:
            return "trend_following"
        elif volatility < 0.5:
            return "range_trading"

        # Time-based sentiment preference (simulate NY hours)
        current_hour = datetime.datetime.utcnow().hour
        if current_hour in [12, 13, 14]:
            return "news_trading"

        if len(df) > 200 and volatility < 1.5:
            return "position_trading"

        return "default"

    except Exception as e:
        logging.warning(f"select_strategy failed: {e}")
        return "default"
