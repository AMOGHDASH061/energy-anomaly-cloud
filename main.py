from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import os

from ml_engine import process_sensor_data

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")
print("DATABASE_URL:", DATABASE_URL)

engine = create_engine(DATABASE_URL)


class SensorData(BaseModel):

    date: str
    time: str
    voltage: float
    current: float
    power: float
    powerfactor: float


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/sensor-data")
def receive_data(data: SensorData):

    # Run ML model
    result = process_sensor_data(
        data.voltage,
        data.current,
        data.power,
        data.powerfactor
    )

    # Insert into Supabase
    query = text("""
        INSERT INTO energy_data(
            datetime,
            voltage,
            current,
            power,
            powerfactor,
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
            :powerfactor,
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
            "voltage": data.voltage,
            "current": data.current,
            "power": data.power,
            "powerfactor": data.powerfactor,
            "power_smooth": result["power_smooth"],
            "delta_power": result["delta_power"],
            "rolling_mean": result["rolling_mean"],
            "rolling_std": result["rolling_std"],
            "anomaly_score": result["anomaly_score"],
            "ml_anomaly": result["ml_anomaly"],
            "power_jump": result["power_jump"],
            "anomaly_detected": result["anomaly_detected"]
        })

        conn.commit()

    return {
        "status": "stored",
        "anomaly": result["anomaly_detected"]
    }
