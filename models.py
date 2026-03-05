from sqlalchemy import Column, Integer, Float, DateTime
from database import Base
from datetime import datetime

class EnergyRecord(Base):
    __tablename__ = "energy_data"

    id = Column(Integer, primary_key=True, index=True)
    consumption = Column(Float)
    prediction = Column(Float)
    anomaly = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)