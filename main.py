# ------------------ Imports ------------------
from fastapi import FastAPI
import numpy as np
import joblib
from collections import deque

from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# ------------------ Load ENV ------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
print("DATABASE_URL =", DATABASE_URL)

# ------------------ DB Setup (Step 4.3) ------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ------------------ DB Table (Step 4.4) ------------------
class EnergyData(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    voltage = Column(Float)
    current = Column(Float)
    power = Column(Float)
    anomaly = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create table in Supabase
Base.metadata.create_all(bind=engine)

# ------------------ FastAPI ------------------
app = FastAPI()

# Load ML model
model = joblib.load("model.pkl")

# ------------------ API Endpoint ------------------
WINDOW_SIZE = 10
power_window = deque(maxlen=WINDOW_SIZE)

@app.post("/predict")
def predict(data: dict):
    voltage = data["voltage"]
    current = data["current"]
    power = data["power"]
    timestamp = data["timestamp"]

    # -------- Feature Engineering --------
    if len(power_window) == 0:
        delta_w = 0.0
    else:
        delta_w = power - power_window[-1]

    power_window.append(power)

    rolling_mean = np.mean(power_window)
    rolling_std = np.std(power_window)

    # -------- ML Feature Vector (ORDER MATTERS) --------
    X = np.array([[power, delta_w, rolling_mean, rolling_std]])

    prediction = int(model.predict(X)[0])   # -1 or 1
    anomaly = 1 if prediction == -1 else 0

    return {
        "timestamp": timestamp,
        "voltage": voltage,
        "current": current,
        "power": power,
        "anomaly": anomaly
    }