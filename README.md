# 🧠 XAU Agent Ensemble AI
AI-based gold trading signal generator using ensemble learning on multi-timeframe data.

## Features
- Multi-timeframe data (daily, weekly, monthly)
- Ensemble model using EMA, ATR, and Random Forest
- Generates BUY/SELL/EXIT signals
- Automatically deployable to Render

## How to run locally
1. python load_data.py
2. python ensemble_train_retrain.py
3. python ensemble_infer_and_signal.py
