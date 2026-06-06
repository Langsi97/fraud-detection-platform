from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    fraud_score = Column(Float, nullable=False)
    prediction = Column(String(32), nullable=False)
    model_version = Column(String(64), nullable=False)
    transaction_amount = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(64), unique=True, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    algorithm = Column(String(64), nullable=False)
    hyperparameters = Column(Text, nullable=True)
    performance_summary = Column(Text, nullable=True)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    fraud_score = Column(Float, nullable=False)
    prediction = Column(String(32), nullable=False)
    model_version = Column(String(64), nullable=False)
    transaction_amount = Column(Float, nullable=True)
    source = Column(String(32), nullable=False, default="api")
