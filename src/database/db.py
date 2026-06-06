import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database.models import Base, ModelMetadata, PredictionLog


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")


def create_tables(database_url: str = None) -> None:
    database_url = database_url or get_database_url()
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)


@contextmanager
def get_session(database_url: str = None):
    database_url = database_url or get_database_url()
    engine = create_engine(database_url, future=True)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def log_prediction_record(session: Session, fraud_score: float, prediction_label: str, model_version: str, transaction_amount: float, source: str = "api") -> None:
    record = PredictionLog(
        fraud_score=fraud_score,
        prediction=prediction_label,
        model_version=model_version,
        transaction_amount=transaction_amount,
        source=source,
    )
    session.add(record)


def log_model_metadata(session: Session, version: str, algorithm: str, hyperparameters: str, performance_summary: str) -> None:
    metadata = ModelMetadata(
        version=version,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        performance_summary=performance_summary,
    )
    session.add(metadata)
