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

    Date: str
    Time: str
    Voltage: float
    Current: float
    Power: float
    Power_Factor: float

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

    voltage = data.Voltage
    current = data.Current
    power = data.Power
    pf = data.Power_Factor

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    normalized_power = Power * (230 / Voltage)

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
            Voltage,
            Current,
            Power,
            Power_Factor,
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
            :Voltage,
            :Current,
            :Power,
            :Power_Factor,
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
            "Voltage": Voltage,
            "Current": Current,
            "Power": Power,
            "Power_Factor": pf,
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
