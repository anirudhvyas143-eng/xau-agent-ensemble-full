🟡 XAUUSD AI Trader (Agentic Ensemble Model)

This project is an AI-driven trading system for XAUUSD (gold) that combines technical indicators, sentiment analysis, and ensemble machine-learning models to generate buy/sell signals. It uses Flask for the backend, Pandas-TA for technical analysis, and machine-learning libraries like LightGBM and Scikit-Learn.

⸻

🚀 Deployment Overview

You can deploy this project instantly on Render.com.
After deployment, your app will be live at:
https://xauusd-ai-trader.onrender.com/signal

⸻

⚙️ Tech Stack
	•	Backend: Flask + Gunicorn
	•	AI Models: LightGBM, Scikit-Learn, Pandas-TA
	•	Python Version: 3.10.13
	•	Hosting: Render (Free plan compatible)
	•	Data Sources: Alpha Vantage and Investing.com

⸻

📁 File Structure

.
├── main.py               → Flask app entry point
├── requirements.txt      → Python dependencies
├── render.yaml           → Render deployment config
├── runtime.txt           → Python version lock (3.10.13)
├── models/               → Trained models
├── data/                 → Cached data
├── logs/                 → Log files
└── README.md             → Project documentation

⸻

🌍 Environment Variables (set in Render Dashboard)
	•	ALPHAV_API_KEY → Your Alpha Vantage API key
	•	RAPID_API_KEY → Optional, for Investing.com API
	•	SELF_PING_URL → Optional, to keep app alive
	•	REFRESH_INTERVAL_SECS → Default: 3600
	•	PORT → Default: 10000

⸻

🧠 Local Setup (Manual Testing)
	1.	Clone repo:
git clone https://github.com/anirudhvyas143-eng/xau-agent-ensemble-full.git
	2.	Navigate to project folder:
cd xau-agent-ensemble-full
	3.	Create virtual environment:
python -m venv venv
	4.	Activate it:
	•	On macOS/Linux: source venv/bin/activate
	•	On Windows: venv\Scripts\activate
	5.	Install dependencies:
pip install -r requirements.txt
	6.	Run app locally:
python main.py
	7.	Visit http://localhost:10000/signal in browser.

⸻

🧰 Render Build Process (render.yaml logic)

Render automatically executes these steps during deployment:
	1.	System setup
	•	apt-get update -y
	•	apt-get install -y gfortran build-essential pkg-config
	2.	Python environment setup
	•	pip install –upgrade pip setuptools wheel
	•	pip install –prefer-binary –no-build-isolation -r requirements.txt
	3.	Folder creation
	•	mkdir -p data models logs
	4.	Start command
	•	gunicorn main:app –workers 1 –threads 2 –timeout 300 –bind 0.0.0.0:$PORT

⸻

🔄 Self-Ping (Keep Alive Feature)

To prevent Render free plan from sleeping, add this snippet before app.run() in main.py:

import threading, requests, time, os

def self_ping():
url = os.getenv(“SELF_PING_URL”)
if not url:
return
while True:
try:
requests.get(url, timeout=10)
print(f”Pinged self URL: {url}”)
except Exception as e:
print(f”Ping error: {e}”)
time.sleep(900)

threading.Thread(target=self_ping, daemon=True).start()

Then, set SELF_PING_URL=https://xauusd-ai-trader.onrender.com/signal in Render environment variables.

⸻

📦 requirements.txt

Flask==3.0.3
Flask-Cors==4.0.0
gunicorn==23.0.0
requests==2.32.3
pandas==2.2.3
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.2
xgboost==2.1.2
lightgbm==4.4.0
optuna==4.0.0
hyperopt==0.2.7
pandas-ta @ git+https://github.com/twopirllc/pandas-ta.git@main

⸻

📜 runtime.txt

python-3.10.13

⸻

🪄 render.yaml

services:
	•	type: web
name: xauusd-ai-trader
env: python
plan: free
region: oregon
autoDeploy: true
buildCommand: |
echo “==== 🛠️ Setting up system dependencies ====”
apt-get update -y
apt-get install -y gfortran build-essential pkg-config
echo “==== 🚀 Installing Python dependencies ====”
pip install –upgrade pip setuptools wheel
pip install –prefer-binary –no-build-isolation -r requirements.txt
mkdir -p data models logs
startCommand: |
gunicorn main:app –workers 1 –threads 2 –timeout 300 –bind 0.0.0.0:$PORT
envVars:
	•	key: PORT
value: 10000
	•	key: REFRESH_INTERVAL_SECS
value: 3600
	•	key: ALPHAV_API_KEY
sync: false
	•	key: RAPID_API_KEY
sync: false
	•	key: SELF_PING_URL
value: https://xauusd-ai-trader.onrender.com/signal

⸻

🧾 License

MIT License © 2025 — Created by Anirudh Vyas

⸻

💡 Notes
	•	Always install pandas-ta from GitHub main branch (@main).
	•	Avoid .whl and .zip URLs — Render doesn’t support them.
	•	Ensure runtime.txt uses Python 3.10.13 for smooth builds.
	•	If build fails due to cache, click “Clear Build Cache” and redeploy from Render dashboard.
