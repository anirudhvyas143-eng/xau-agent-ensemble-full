# 🦾 XAUUSD AI Trader Agent — Adaptive ML + RL + Agentic Automation

A next-gen, full-stack AI-powered trading engine for Gold (XAUUSD).  
This system merges Machine Learning, Reinforcement Learning, Quantitative Analysis, and Agentic AI automation to deliver real-time BUY/SELL signals on hourly and daily timeframes — retraining itself automatically.

**Important:** This build *does not* use yfinance. Data sources: Investing.com (via RapidAPI) + AlphaVantage FX intraday.

## Quick Start (Render)
1. Push repo to GitHub.
2. On Render, create a Web Service and connect the repo.
3. Set environment variables (ALPHAV_API_KEY, RAPIDAPI_KEY).
4. Deploy (Render will run `pip install -r requirements.txt` and start the app).

## API Endpoints
- `GET /` — health
- `GET /signal` — run fetch/train/infer and return combined daily/hourly signals
- `GET /history` — recent signals
- `GET /dashboard` — simple HTML dashboard

## Files
- `main.py` — primary Flask app and pipeline (Investing + AlphaVantage)
- `load_data.py` — data fetch helpers
- `features_engineering.py` — indicator calculations
- `strategy_manager.py` — multi-strategy glue (trend, range, news placeholders)
- `ensemble_*` — placeholders for ensemble training/inference
- `ai_automation.py` — (light) automation wrapper to trigger retrain
- `backtest_engine.py` — simple backtesting harness
- `render.yaml`, `requirements.txt`, `.python-version` — Render settings

## License
MIT © 2025 Anirudh Vyas


