# 🦾 XAUUSD AI Trader Agent  
AI-powered Gold (XAUUSD) signal generator that analyzes historical and live data, trains a Random Forest ML model, and generates BUY/SELL signals every 4 hours — fully deployable on Render’s free web service.

---

## ⚙️ Features  
✅ Uses TradingView historical CSVs (daily, weekly, monthly)  
✅ Generates AI-powered indicator set (EMA, RSI, ATR, Volatility, Momentum)  
✅ Trains Random Forest model on multi-timeframe features  
✅ Predicts next 4-hour price movement for XAUUSD  
✅ Outputs AI signals with **conservative**, **aggressive**, and **safer** entries  
✅ Fully automated refresh and deployment  
✅ Flask web API for live integration with bots, dashboards, or alerts  

---

## 📂 Project Directory Structure  
```
/data                  → CSV historical datasets (daily, weekly, monthly)  
/models                → rf_model.joblib (auto-generated after training)  
/main.py               → Core AI trading logic file  
/requirements.txt      → Python dependencies list  
/render.yaml           → Render web service configuration  
/README.md             → Project overview, API documentation, and setup guide  
```

---

## 🚀 Deployment on Render  

1️⃣ Push this repo to **GitHub**  
2️⃣ Visit [Render.com](https://render.com) → “New Web Service”  
3️⃣ Select your GitHub repository  
4️⃣ **Choose Free Plan** and click **Deploy**  
5️⃣ **Build & Start Commands:**
```bash
pip install -r requirements.txt
python main.py
```

6️⃣ Example `render.yaml`:
```yaml
services:
  - type: web
    name: xauusd-ai-trader
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python main.py"
    plan: free
    envVars:
      - key: INFER_INTERVAL_SECS
        value: 14400  # every 4 hours
```

---

## 🌐 API Endpoints  

### 🔹 1. Health Check  
**GET** `https://<your-app-name>.onrender.com/`  
**Response:**
```json
{
  "status": "ok",
  "time": "2025-10-28T18:30:00Z",
  "message": "XAUUSD AI Trader active and monitoring"
}
```

---

### 🔹 2. Run AI Signal Generation  
**GET** `https://<your-app-name>.onrender.com/run`  
**Response:**
```json
{
  "status": "ok",
  "result": {
    "timestamp": "2025-10-28T18:32:15Z",
    "signal": "BUY",
    "confidence": 0.87,
    "conservative_entry": 2378.2,
    "aggressive_entry": 2385.7,
    "safer_entry": 2375.6,
    "take_profit": 2398.0,
    "stop_loss": 2365.0
  }
}
```

---

### 🔹 3. Historical Signal Log  
**GET** `https://<your-app-name>.onrender.com/history`  
**Response:**
```json
{
  "history": [
    {"time": "2025-10-27T14:00Z", "signal": "BUY", "price": 2372.4},
    {"time": "2025-10-27T18:00Z", "signal": "HOLD", "price": 2381.1},
    {"time": "2025-10-28T02:00Z", "signal": "SELL", "price": 2369.7}
  ]
}
```

---

## 🧠 Entry Logic  
| Entry Type   | Description             | Strategy Use          |
|---------------|-------------------------|------------------------|
| Aggressive    | High-risk, high-reward  | Intraday / Scalping   |
| Conservative  | Balanced entry          | Swing trading         |
| Safer         | Low-risk confirmation   | Position trades       |

---

## 🕒 Keep Alive  
Use **[UptimeRobot](https://uptimerobot.com)** to ping your app every 15 minutes:  
`https://<your-app-name>.onrender.com/`

---

## 📜 License  
**MIT License © 2025 – Anirudh Vyas**

---

## 🧩 Usage Notes  
- Default interval: **4 hours** (`INFER_INTERVAL_SECS` in Render dashboard).  
- CSV data must exist in `/data` before first run.  
- Auto-retrains model when `/run` detects new data.  
- Logs signals automatically to `/history`.  

---

## 🧾 API Summary  
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Health check |
| `/run` | GET/POST | Generate new AI signals |
| `/history` | GET | Retrieve past signals |

---

## 👨‍💻 Author  
**Developed by:** Anirudh Vyas  
**Year:** 2025  
**Purpose:** AI-powered quantitative trading system for accurate and continuous XAUUSD signal generation.







