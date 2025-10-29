🦾 XAUUSD AI Trader Agent — Adaptive ML + RL + Automation

A next-gen, full-stack AI-powered trading engine for Gold (XAUUSD).
This system merges Machine Learning, Reinforcement Learning, Quantitative Analysis, and Agentic AI automation to deliver real-time BUY/SELL signals on both hourly and daily timeframes — retraining itself automatically to stay profitable in any market condition.

⸻

⚙️ Core Capabilities
✅ Dual-Timeframe Signal Engine — Hourly (1H) and Daily (1D)
✅ Automated Data Pipeline — auto-fetches & cleans gold price data
✅ Feature Engineering Suite — EMA, ATR, RSI, Volatility, Momentum, Regime Detection
✅ ML + RL Hybrid Models — ensemble of XGBoost, LightGBM, CatBoost + Stable-Baselines (PPO/DQN)
✅ AI Automation Layer — detects drift, retrains, and redeploys without manual input
✅ Optuna + HyperOpt Optimization — parameter tuning for peak signal precision
✅ Backtesting + QuantStats Reports — transparent strategy metrics & equity curves
✅ REST API + Dashboard Ready — simple endpoints for bots, dashboards, or alerts
✅ Render Free-Plan Compatible — deploy instantly with hourly & daily refresh

⸻

📂 Project Structure

/data — Historical CSV datasets (hourly, daily, weekly)
/models — Auto-saved ML / RL models
/logs — Runtime and retraining logs
/features_engineering.py — Feature generation (EMA, RSI, ATR, Momentum)
/train_model.py — Initial supervised + ensemble model training
/ensemble_train_retrain.py — Continuous retraining + ensemble logic
/ai_automation.py — Drift detection, auto-retrain, version control
/backtest_engine.py — Performance simulation + metrics + plots
/main.py — Core Flask API & hourly/daily inference logic
/requirements.txt — Dependencies
/render.yaml — Render deploy configuration
/README.md — Documentation (this file)

⸻

🚀 One-Click Render Deployment
	1.	Push this repo to GitHub:

git init
git add .
git commit -m “Initial XAUUSD AI Trader commit”
git branch -M main
git remote add origin https://github.com/yourname/xauusd-ai-trader.git
git push -u origin main
	2.	Go to Render.com → New Web Service
	3.	Connect your GitHub repo
	4.	Choose Free Plan

Build Command:
pip install –upgrade pip
pip install -r requirements.txt

Start Command:
python main.py

⸻

🧠 Smart AI Automation Overview

features_engineering.py → Generates EMA, RSI, ATR, Momentum, Volatility, Z-Score features
train_model.py → Trains baseline XGBoost/LightGBM/CatBoost + RL models
ensemble_train_retrain.py → Periodically retrains ensemble → saves best performer
ai_automation.py → Detects model drift → auto-retrain → push new model to Render
backtest_engine.py → Compares AI vs Market, exports metrics + plot
main.py → Flask API: /run, /run_daily, /history

⸻

🌐 API Endpoints

Health Check
GET /
Response:
{ “status”: “ok”, “time”: “2025-10-29T06:00Z”, “message”: “XAUUSD AI Trader active (hourly + daily)” }

Run Hourly Signal
GET /run
Response:
{ “status”: “ok”, “result”: { “timestamp”: “2025-10-29T06:02Z”, “timeframe”: “1H”, “signal”: “BUY”, “confidence”: 0.93, “conservative_entry”: 2382.4, “aggressive_entry”: 2386.1, “safer_entry”: 2379.8, “take_profit”: 2401.2, “stop_loss”: 2368.9 } }

Run Daily Signal
GET /run_daily
Response:
{ “status”: “ok”, “result”: { “timestamp”: “2025-10-29T06:05Z”, “timeframe”: “1D”, “signal”: “SELL”, “confidence”: 0.88 } }

Signal History
GET /history
Response:
{ “history”: [ { “time”: “2025-10-28T20:00Z”, “tf”: “1H”, “signal”: “BUY”, “price”: 2378.3 }, { “time”: “2025-10-28”, “tf”: “1D”, “signal”: “SELL”, “price”: 2369.5 } ] }

⸻

📊 Backtesting & Performance

backtest_engine.py runs simulated trades using historical data.
Generates:
	•	strategy_performance.png → AI vs Market growth curve
	•	backtest_results.csv → detailed log
	•	Key metrics: Total Return %, Sharpe Ratio, Max Drawdown

⸻

🧩 AI Tech Stack

Machine Learning: XGBoost, LightGBM, CatBoost
Reinforcement Learning: Stable-Baselines (PPO / DQN)
Optimization: Optuna, HyperOpt
Feature Analysis: pandas-ta, ta, NumPy, SciPy
Backtesting & Reports: Backtrader, QuantStats, Matplotlib
Automation & Orchestration: ai_automation.py + Render env loops

⸻

🕒 Keep Alive Tip

Add an UptimeRobot ping every 15 min to keep Render service awake:
https://xauusd-ai-trader.onrender.com/

⸻

🧾 API Summary

/ — GET — Health check
/run — GET — Generate hourly signal
/run_daily — GET — Generate daily signal
/history — GET — Fetch recent signals

⸻

📜 License

MIT License © 2025 – Anirudh Vyas

⸻

👨‍💻 Author

Developed by: Anirudh Vyas
Purpose: Build a self-learning AI quant ecosystem for precision XAUUSD trading using ML + RL automation.






