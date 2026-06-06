# Credit Card Fraud Detection Platform

A production-style credit card fraud detection platform built with Python, MLflow, FastAPI, SQLite, Evidently, and Tableau-ready exports.

## Business Problem
Credit card fraud is a high-cost risk for digital payments. This platform detects fraudulent transactions in real time, monitors model drift, and exports business-ready summaries for BI dashboards.

## Architecture
- `data/raw/`: original dataset storage
- `data/processed/`: cleaned and validated CSV outputs
- `src/data/`: dataset loading and validation
- `src/models/`: training, evaluation, and prediction logic
- `src/api/`: FastAPI application and schemas
- `src/database/`: SQLite ORM models and database utilities
- `src/monitoring/`: drift detection and metrics tracking
- `src/pipelines/`: training and batch prediction orchestration
- `reports/`: EDA and monitoring outputs
- `tableau/dashboard_data/`: BI-ready exports

![Architecture diagram](screenshots/architecture.png)

## Dataset
The project uses the OpenML credit card fraud dataset in ARFF format.
- Target column: `Class`
- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Modeling Approach
- Baseline: Logistic Regression
- Ensemble: Random Forest
- Primary model: XGBoost
- Class imbalance handled with class weighting
- Evaluation based on PR-AUC, ROC-AUC, Precision, Recall, F1 score

## Getting Started

### Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Train the models
```bash
python src/pipelines/training_pipeline.py
```

### Run the API locally
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Start with Docker
```bash
docker-compose up --build
```

## API Endpoints
- `GET /` — service metadata
- `GET /health` — health check
- `POST /predict` — single transaction prediction
- `POST /batch-predict` — batch CSV prediction

## Monitoring and BI
- Evidently monitoring report saved to `reports/monitoring/`
- Tableau-ready exports saved to `tableau/dashboard_data/`

## Future Improvements
- add automated retraining based on drift alerts
- support feature store integration
- integrate a production-grade database such as PostgreSQL
- add authentication and rate limiting to the API
