from fastapi import FastAPI

from schemas import (
    SessionInput,
    PredictionResponse
)

from predictor import predict_session


app = FastAPI(
    title="E-commerce Conversion Intelligence API",
    description=(
        "Predicts purchase propensity from the "
        "first five minutes of session behaviour."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "E-commerce Conversion Intelligence API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest",
        "calibration": "Isotonic Regression",
        "observation_window": "First 5 minutes",
        "prediction_target": "Purchase after minute 5"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    session: SessionInput
):
    return predict_session(session)