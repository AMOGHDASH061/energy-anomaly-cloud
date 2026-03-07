from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
import joblib
import os
from collections import deque

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

# -----------------------------
# LOAD ML COMPONENTS
# -----------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
delta_threshold = joblib.load("delta_threshold.pkl")

# -----------------------------
# FASTAPI APP
# -----------------------------

app = FastAPI(title="Energy Audit AI API")

# -----------------------------
# INPUT SCHEMA (ESP32)
# -----------------------------

class SensorData(BaseModel):

    datetime: str
    voltage: float
    current: float
    power: float
    power_factor: float

# -----------------------------
# SLIDING WINDOW STATE
# -----------------------------

WINDOW_SIZE = 10
power_window = deque(maxlen=WINDOW_SIZE)

last_power = None

# -----------------------------
# HOME ENDPOINT
# -----------------------------

@app.get("/")
def home():
    return {"status": "Energy AI API running"}

# -----------------------------
# SENSOR DATA ENDPOINT
# -----------------------------

@app.post("/sensor-data")
def receive_data(data: SensorData):

    global last_power

    voltage = data.voltage
    current = data.current
    power = data.power
    pf = data.power_factor

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    normalized_power = power * (230 / voltage)

    power_window.append(normalized_power)

    power_smooth = np.median(list(power_window)[-2:]) if len(power_window) >= 2 else normalized_power

    if last_power is None:
        delta_power = 0
    else:
        delta_power = power_smooth - last_power

    last_power = power_smooth

    rolling_mean = np.mean(power_window)

    rolling_std = np.std(power_window)

    # -----------------------------
    # ML MODEL INPUT
    # -----------------------------

    features = [[
        power_smooth,
        delta_power,
        rolling_mean,
        rolling_std,
        current,
        pf
    ]]

    X_test = scaler.transform(features)

    score = model.decision_function(X_test)[0]

    pred = model.predict(X_test)[0]

    ml_anomaly = 1 if pred == -1 else 0

    # -----------------------------
    # POWER JUMP DETECTION
    # -----------------------------

    power_jump = 1 if abs(delta_power) > delta_threshold else 0

    anomaly_detected = 1 if (ml_anomaly or power_jump) else 0

    # -----------------------------
    # STORE TO SUPABASE
    # -----------------------------

    query = text("""
        INSERT INTO energy_data(
            datetime,
            voltage,
            current,
            power,
            power_factor,
            power_smooth,
            delta_power,
            rolling_mean,
            rolling_std,
            anomaly_score,
            ml_anomaly,
            power_jump,
            anomaly_detected
        )
        VALUES(
            :datetime,
            :voltage,
            :current,
            :power,
            :power_factor,
            :power_smooth,
            :delta_power,
            :rolling_mean,
            :rolling_std,
            :anomaly_score,
            :ml_anomaly,
            :power_jump,
            :anomaly_detected
        )
    """)

    with engine.connect() as conn:

        conn.execute(query, {
            "datetime": data.datetime,
            "voltage": voltage,
            "current": current,
            "power": power,
            "power_factor": pf,
            "power_smooth": power_smooth,
            "delta_power": delta_power,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "anomaly_score": float(score),
            "ml_anomaly": ml_anomaly,
            "power_jump": power_jump,
            "anomaly_detected": anomaly_detected
        })

        conn.commit()

    return {
        "status": "stored",
        "anomaly": anomaly_detected
    }
