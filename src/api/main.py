import os
from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResponse, TransactionRequest
from src.database.db import create_tables, get_session, log_prediction_record
from src.models.predict import load_model, load_preprocessor, predict_batch, predict_transaction


MODEL_PATH = os.getenv("MODEL_PATH", "./models/best_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "./models/preprocessor.pkl")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")

model = None
preprocessor = None

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict credit card fraud risk and track prediction metadata in SQLite.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    global model, preprocessor
    create_tables(DATABASE_URL)
    model = load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "service": "Credit Card Fraud Detection",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/health", response_class=JSONResponse)
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    try:
        result = predict_transaction(model, preprocessor, request.transaction)

        with get_session(DATABASE_URL) as session:
            log_prediction_record(
                session=session,
                fraud_score=result["fraud_probability"],
                prediction_label=result["prediction"],
                model_version="production",
                transaction_amount=request.transaction.get("Amount", 0.0),
            )

        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            prediction=result["prediction"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=415, detail="CSV file required")
    try:
        df = pd.read_csv(file.file)
        predictions = predict_batch(model, preprocessor, df)

        with get_session(DATABASE_URL) as session:
            for _, row in predictions.iterrows():
                log_prediction_record(
                    session=session,
                    fraud_score=float(row["fraud_probability"]),
                    prediction_label=row["prediction"],
                    model_version="production",
                    transaction_amount=float(row.get("Amount", 0.0)),
                )

        response_data = [
            PredictionResponse(
                fraud_probability=float(row["fraud_probability"]),
                prediction=row["prediction"],
            )
            for _, row in predictions.iterrows()
        ]

        return BatchPredictionResponse(predictions=response_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
