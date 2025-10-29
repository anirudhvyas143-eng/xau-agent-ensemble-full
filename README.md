🦾 XAUUSD AI Trader Agent — Hourly + Daily Hybrid

Full-stack AI-powered Gold (XAUUSD) signal generator combining advanced ML, RL, and quantitative finance.
Automatically trains, optimizes, backtests, and deploys BUY/SELL signals every hour and day — fully hosted on Render’s Free Plan.

⸻

⚙️ Features

✅ Dual-Timeframe Prediction — Hourly & Daily Signals
✅ Advanced Feature Engineering (EMA, ATR, RSI, Momentum, Volatility)
✅ Automated Plan Optimization & Backtesting Loop
✅ Financial Metrics + QuantStats Reports
✅ Unsupervised Clustering for Market Regime Detection
✅ Deep Reinforcement Learning for Adaptive Entry Logic
✅ Model Drift Detection & Auto-Retraining
✅ Hyperparameter Tuning with Optuna / HyperOpt
✅ Flask Web API for bots, dashboards, or alerts
✅ Deployed on Render with hourly and daily refresh automation

⸻

📂 Project Structure

/data → Historical datasets (CSV: hourly, daily, weekly)
/models → Saved ML and RL models (auto-generated)
/main.py → Core AI trading logic
/requirements.txt → Dependency list
/render.yaml → Render deploy configuration
/README.md → Documentation (this file)

⸻

🚀 One-Click Render Deployment

1️⃣ Push this repo to GitHub
2️⃣ Go to Render.com → “New Web Service”
3️⃣ Select your GitHub repository
4️⃣ Choose Free Plan and click Deploy
5️⃣ Build & Start commands:

pip install -r requirements.txt
python main.py

6️⃣ Example render.yaml:

services:
	•	type: web
name: xauusd-ai-trader
env: python
buildCommand: “pip install -r requirements.txt”
startCommand: “python main.py”
plan: free
envVars:
	•	key: INFER_INTERVAL_SECS
value: 3600  # every 1 hour
	•	key: DAILY_REFRESH_SECS
value: 86400  # every 1 day

⸻

🌐 API Endpoints
	1.	Health Check
GET → https://.onrender.com/
Response:
{
“status”: “ok”,
“time”: “2025-10-29T06:00Z”,
“message”: “XAUUSD AI Trader active (hourly + daily)”
}
	2.	Run AI Signal Generation
GET → https://.onrender.com/run
Response:
{
“status”: “ok”,
“result”: {
“timestamp”: “2025-10-29T06:02Z”,
“timeframe”: “1H”,
“signal”: “BUY”,
“confidence”: 0.92,
“conservative_entry”: 2382.4,
“aggressive_entry”: 2386.1,
“safer_entry”: 2379.8,
“take_profit”: 2401.2,
“stop_loss”: 2368.9
}
}
	3.	Daily Signal Run
GET → https://.onrender.com/run_daily
Response:
{
“status”: “ok”,
“result”: {
“timestamp”: “2025-10-29T06:05Z”,
“timeframe”: “1D”,
“signal”: “SELL”,
“confidence”: 0.88
}
}
	4.	Signal History
GET → https://.onrender.com/history
Response:
{
“history”: [
{“time”: “2025-10-28T20:00Z”, “tf”: “1H”, “signal”: “BUY”, “price”: 2378.3},
{“time”: “2025-10-28”, “tf”: “1D”, “signal”: “SELL”, “price”: 2369.5}
]
}

⸻

🧠 Entry Logic

Aggressive → High risk / high reward (scalping, intraday)
Conservative → Balanced (swing trading)
Safer → Low risk (position trades)

⸻

🧩 AI Modules Used

• Automated Plan Optimization — Optuna + RL fine-tuning
• Backtesting & Metrics — Backtrader + QuantStats
• Feature Engineering — EMA, RSI, ATR, Volatility, Momentum
• Unsupervised Learning — KMeans for Market Regime Detection
• Reinforcement Learning — Stable-Baselines3 (PPO/DQN)
• Hyperparameter Tuning — Optuna / HyperOpt
• Model Drift Detection — Auto-retrain when accuracy drops
• Exploratory Data Analysis — Automatic EDA before retraining

⸻

🕒 Keep Alive

Use UptimeRobot to ping your app every 15 minutes:
https://.onrender.com/

⸻

🧾 API Summary

/ → GET → Health check
/run → GET → Generate hourly signal
/run_daily → GET → Generate daily signal
/history → GET → Retrieve signal history

⸻

📜 License

MIT License © 2025 – Anirudh Vyas

⸻

👨‍💻 Author

Developed by: Anirudh Vyas
Purpose: Next-gen AI quant ecosystem for precision XAUUSD signal generation — hourly and daily, optimized via ML + RL automation.






