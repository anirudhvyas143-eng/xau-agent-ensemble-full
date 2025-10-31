# strategy_manager.py — combine multiple strategy signals into ensemble weights
import numpy as np

def trend_following_signal(df):
    """
    Simple EMA crossover trend-following:
    if ema8 > ema21 -> bullish; if ema8 < ema21 -> bearish
    returns +1 (buy), -1 (sell), 0 (neutral)
    """
    if df is None or len(df) < 2:
        return 0
    last = df.iloc[-1]
    if last["ema8"] > last["ema21"]:
        return 1
    if last["ema8"] < last["ema21"]:
        return -1
    return 0

def range_signal(df, window=50):
    """
    Simple range detection: if price near top of range -> sell; near bottom -> buy
    """
    if df is None or len(df) < window:
        return 0
    r = df["Close"].tail(window)
    top = r.max()
    bot = r.min()
    last = r.iloc[-1]
    pct = (last - bot) / (top - bot + 1e-9)
    if pct > 0.9:
        return -1
    if pct < 0.1:
        return 1
    return 0

def news_bias_signal(headline_sentiment_score):
    """
    Placeholder — if integrated with news sentiment, convert to +1/-1/0
    """
    if headline_sentiment_score is None:
        return 0
    if headline_sentiment_score > 0.2:
        return 1
    if headline_sentiment_score < -0.2:
        return -1
    return 0

def ensemble_decision(signals, weights=None):
    """
    signals: dict like {'trend':1, 'range':-1, 'news':0}
    weights: dict of weights
    returns combined score (-1..1) and discrete decision
    """
    if weights is None:
        weights = {k:1.0 for k in signals.keys()}
    total = 0.0
    wsum = 0.0
    for k,v in signals.items():
        w = weights.get(k, 1.0)
        total += w * v
        wsum += abs(w)
    score = (total / (wsum + 1e-12))
    if score > 0.2:
        return {"score": score, "decision": "BUY"}
    if score < -0.2:
        return {"score": score, "decision": "SELL"}
    return {"score": score, "decision": "NEUTRAL"}
