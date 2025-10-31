# 🦾 XAUUSD AI Trader Agent — Adaptive ML + RL + Agentic Automation

A next-gen, full-stack AI-powered trading engine for Gold (XAUUSD).  
This system merges Machine Learning, Reinforcement Learning, Quantitative Analysis, and Agentic AI automation to deliver real-time BUY/SELL signals on both hourly and daily timeframes — retraining itself automatically to stay profitable in any market condition.

---

## ⚙️ Core Capabilities
✅ Dual-Timeframe Signal Engine — Hourly (1H) and Daily (1D)  
✅ Automated Data Pipeline — Auto-fetches & cleans gold price data via Yahoo Finance  
✅ Feature Engineering Suite — EMA, ATR, RSI, Volatility, Momentum, Regime Detection  
✅ Hybrid ML + RL Models — XGBoost, LightGBM, CatBoost + Stable-Baselines (PPO/DQN)  
✅ Self-Learning AI Automation Layer — Detects drift, retrains, redeploys automatically  
✅ Optuna + HyperOpt Optimization — Peak signal precision tuning  
✅ Backtesting + QuantStats Reports — Transparent equity curves & strategy metrics  
✅ REST API + Dashboard Ready — Endpoints for trading bots, dashboards, or alerts  
✅ Render Free-Plan Compatible — Auto-refresh with hourly & daily inference  

---

## 🧩 New Multi-Strategy Engine (Added Oct 2025)
Integrated Fundamental + Technical Strategy Fusion:
- Trend-Following — Trades with macro direction using multi-EMA crossovers  
- News Trading — Adapts to volatility surges post major events (CPI, Fed, geopolitical)  
- Range Trading — Detects support/resistance ranges & mean-reversion setups  
- Position Trading — Long-term, fundamentally driven signal bias  
- Auto-Ensemble Blend — Dynamic weighting between fundamental + technical signals  

Controlled via `strategy_manager.py`, allowing plug-and-play custom strategies and weighting.

---

## 📂 Project Structure
/data — Historical datasets (auto-fetched from Yahoo Finance)  
/models — Auto-saved ML & RL model checkpoints  
/logs — Runtime, drift, and retraining logs  
/load_data.py — Fetch + preprocess multi-timeframe XAUUSD data  
/features_engineering.py — Technical feature builder (EMA, RSI, ATR, Momentum)  
/strategy_manager.py — Core fusion: trend-following, range, news, position trading  
/ensemble_train_retrain.py — Continuous retraining & ensemble logic  
/ai_automation.py — Drift detection + auto-retrain pipeline  
/backtest_engine.py — Strategy simulation + QuantStats reports  
/main.py — Flask API (inference & signal serving)  
/requirements.txt — Dependencies  
/render.yaml — Render deploy configuration  
/README.md — Documentation (this file)

---

## 🚀 Quick Deploy on Render
1️⃣ Push this repo to GitHub  
git init  
git add .  
git commit -m "Initial XAUUSD AI Trader commit"  
git branch -M main  
git remote add origin https://github.com/yourname/xauusd-ai-trader.git  
git push -u origin main  

2️⃣ Go to Render.com → “New Web Service”  
- Connect your GitHub repo  
- Choose Free Plan  

Build Command:  
pip install --upgrade pip  
pip install --prefer-binary --no-build-isolation -r requirements.txt  

Start Command:  
python ai_automation.py  

---

## 🧠 System Flow Overview
load_data.py → Fetches gold data (daily/weekly/monthly) via Yahoo Finance  
features_engineering.py → Computes EMA, RSI, ATR, Momentum, Volatility, etc.  
strategy_manager.py → Blends multiple strategies (trend/news/range/position)  
ensemble_train_retrain.py → Periodic retraining & model selection  
ai_automation.py → Monitors drift, triggers retraining, logs updates  
backtest_engine.py → Generates performance reports  
main.py → Flask API endpoints for signals & history  

---

## 🌐 API Endpoints
**Health Check:**  
GET /  
→ { "status": "ok", "time": "2025-10-29T06:00Z", "message": "XAUUSD AI Trader active" }

**Run Hourly Signal:**  
GET /run  
→ { "signal": "BUY", "confidence": 0.93, "entry": 2382.4, "tp": 2401.2, "sl": 2368.9 }

**Run Daily Signal:**  
GET /run_daily  
→ { "signal": "SELL", "confidence": 0.88 }

**Signal History:**  
GET /history  
→ { "history": [{ "tf": "1H", "signal": "BUY", "price": 2378.3 }, ...] }

---

## 📊 Backtesting & Performance
Run backtesting:  
python backtest_engine.py  

Outputs:  
- strategy_performance.png → AI vs Market equity curve  
- backtest_results.csv → full trade log  
- Metrics: Total Return %, Sharpe, Sortino, Max Drawdown  

---

## 🧰 AI Tech Stack
| Layer | Tools |
|-------|-------|
| Data Source | Yahoo Finance (yfinance) |
| Feature Engine | pandas-ta, ta, NumPy, SciPy |
| Machine Learning | XGBoost, LightGBM, CatBoost |
| Reinforcement Learning | Stable-Baselines (PPO / DQN) |
| Optimization | Optuna, HyperOpt |
| Backtesting & Reports | Backtrader, QuantStats, Matplotlib |
| Automation | ai_automation.py + Render auto-deploy |
| API | Flask (REST) |

---

## 🕒 Keep Alive (Optional)
Add a free UptimeRobot ping every 15 minutes:  
https://xauusd-ai-trader.onrender.com/

---

## 🧾 API Summary
| Endpoint | Method | Description |
|-----------|---------|-------------|
| / | GET | Health check |
| /run | GET | Generate hourly signal |
| /run_daily | GET | Generate daily signal |
| /history | GET | Fetch recent signal logs |

---

## 🧠 Future Expansion
- Multi-asset support (XAGUSD, EURUSD, BTCUSD)  
- News sentiment ingestion (Twitter, Bloomberg, Fed statements)  
- Agentic portfolio management  
- MetaTrader 5 / cTrader webhook integration  

---

## 📜 License
MIT License © 2025 – Anirudh Vyas 

---

## 👨‍💻 Author
Developed by: **Anirudh Vyas**  
Purpose: Build a self-learning, adaptive AI quant ecosystem for precision XAUUSD trading — blending ML, RL, and Agentic AI automation.


