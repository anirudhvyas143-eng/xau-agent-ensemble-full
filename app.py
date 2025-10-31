# app.py - optional smaller API wrapper that imports from main
from main import app
# Expose same Flask app; this file exists if you want gunicorn to use `app:app`
