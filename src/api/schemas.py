from typing import Dict, List

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction: Dict[str, float] = Field(
        ..., description="Transaction feature vector keyed by feature name"
    )


class BatchPredictionRequest(BaseModel):
    transactions: List[Dict[str, float]] = Field(
        ..., description="A list of transactions containing feature values"
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
