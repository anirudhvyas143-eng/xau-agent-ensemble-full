# ai_automation.py — light automation loop (used if you want a separate entrypoint)
import time, json
from main import build_train_and_signal
from drift_manager import psi

def run_loop(interval=3600):
    while True:
        combined = build_train_and_signal()
        # lightweight drift check: compare last two confidences
        try:
            hist = json.load(open("signals_history.json"))
            if len(hist) >= 2:
                prev = hist[-2]
                curr = hist[-1]
                # naive psi between daily confidences
                prev_conf = [prev.get("daily",{}).get("confidence",0)]
                curr_conf = [curr.get("daily",{}).get("confidence",0)]
                score = psi(prev_conf, curr_conf)
                if score > 0.5:
                    print("⚠️ Drift detected (simple PSI) — consider retrain or alert")
        except Exception:
            pass
        time.sleep(interval)

if __name__ == "__main__":
    run_loop()
