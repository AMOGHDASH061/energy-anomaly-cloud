import numpy as np
import joblib
from collections import deque

# Load ML components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
delta_threshold = joblib.load("delta_threshold.pkl")

WINDOW_SIZE = 10
power_window = deque(maxlen=WINDOW_SIZE)

last_power = None


def process_sensor_data(Voltage, Current, Power, Power_Factor):

    global last_power

    # Voltage normalization
    normalized_power = power * (230 / Voltage)

    power_window.append(normalized_power)

    # Smoothing
    if len(power_window) >= 2:
        power_smooth = np.median(list(power_window)[-2:])
    else:
        power_smooth = normalized_power

    # Delta power
    if last_power is None:
        delta_power = 0
    else:
        delta_power = power_smooth - last_power

    last_power = power_smooth

    # Rolling features
    rolling_mean = np.mean(list(power_window)[-3:])
    rolling_std = np.std(list(power_window)[-3:])

    features = [[
        power_smooth,
        delta_power,
        rolling_mean,
        rolling_std,
        Current,
        Power_Factor
    ]]

    X_test = scaler.transform(features)

    score = model.decision_function(X_test)[0]

    pred = model.predict(X_test)[0]

    ml_anomaly = 1 if pred == -1 else 0

    power_jump = 1 if abs(delta_power) > delta_threshold else 0

    anomaly_detected = 1 if (ml_anomaly or power_jump) else 0

    return {
        "power_smooth": power_smooth,
        "delta_power": delta_power,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "anomaly_score": float(score),
        "ml_anomaly": ml_anomaly,
        "power_jump": power_jump,
        "anomaly_detected": anomaly_detected
    }
