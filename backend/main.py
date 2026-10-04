from fastapi import FastAPI

from .schemas import (
    SessionInput,
    PredictionResponse
)

from .predictor import (
    predict_session
)


app = FastAPI(
    title=(
        "E-commerce Conversion "
        "Propensity API"
    ),
    version="1.1.0",
    description=(
        "Predicts purchase propensity "
        "from the first five minutes "
        "of e-commerce session behavior "
        "and returns live SHAP explanations."
    )
)


@app.get("/")
def root():

    return {
        "message":
            (
                "E-commerce Conversion "
                "Propensity API"
            ),

        "docs":
            "/docs"
    }


@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


@app.get("/model-info")
def model_info():

    return {
        "model":
            "Random Forest",

        "calibration":
            "Isotonic Regression",

        "observation_window":
            "First 5 minutes",

        "prediction_target":
            "Purchase after minute 5",

        "explainability":
            "Live SHAP"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    session: SessionInput
):

    return predict_session(
        session
    )