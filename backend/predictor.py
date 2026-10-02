import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

CALIBRATOR_PATH = (
    BASE_DIR
    / "models"
    / "isotonic_calibrator.pkl"
)

CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "deployment_config.json"
)


rf_model = joblib.load(MODEL_PATH)

calibrator = joblib.load(
    CALIBRATOR_PATH
)

with open(CONFIG_PATH) as f:
    config = json.load(f)


HIGH_THRESHOLD = config[
    "high_intent_raw_score_threshold"
]

MEDIUM_THRESHOLD = config[
    "medium_intent_raw_score_threshold"
]


FEATURE_COLUMNS = [
    "total_events_5min",
    "page_views_5min",
    "product_views_5min",
    "product_clicks_5min",
    "add_to_cart_5min",
    "begin_checkout_5min",
    "search_events_5min",
    "is_returning_user",
    "has_cart",
    "has_checkout",
    "has_product_view",
    "has_product_click",
    "product_views_per_page",
    "click_per_product_view",
    "cart_per_product_view",
    "is_weekend",
    "device_category",
    "operating_system",
    "traffic_source",
    "traffic_medium"
]


def safe_ratio(numerator, denominator):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_features(payload):

    session_date = pd.Timestamp(
        payload.session_date
    )

    row = {
        "total_events_5min":
            payload.total_events_5min,

        "page_views_5min":
            payload.page_views_5min,

        "product_views_5min":
            payload.product_views_5min,

        "product_clicks_5min":
            payload.product_clicks_5min,

        "add_to_cart_5min":
            payload.add_to_cart_5min,

        "begin_checkout_5min":
            payload.begin_checkout_5min,

        "search_events_5min":
            payload.search_events_5min,

        "is_returning_user":
            1 - payload.is_new_user,

        "has_cart":
            int(payload.add_to_cart_5min > 0),

        "has_checkout":
            int(payload.begin_checkout_5min > 0),

        "has_product_view":
            int(payload.product_views_5min > 0),

        "has_product_click":
            int(payload.product_clicks_5min > 0),

        "product_views_per_page":
            safe_ratio(
                payload.product_views_5min,
                payload.page_views_5min
            ),

        "click_per_product_view":
            safe_ratio(
                payload.product_clicks_5min,
                payload.product_views_5min
            ),

        "cart_per_product_view":
            safe_ratio(
                payload.add_to_cart_5min,
                payload.product_views_5min
            ),

        "is_weekend":
            int(session_date.dayofweek >= 5),

        "device_category":
            payload.device_category,

        "operating_system":
            payload.operating_system,

        "traffic_source":
            payload.traffic_source,

        "traffic_medium":
            payload.traffic_medium
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )


def get_intent_segment(raw_score):

    if raw_score >= HIGH_THRESHOLD:
        return "High Intent"

    if raw_score >= MEDIUM_THRESHOLD:
        return "Medium Intent"

    return "Low Intent"


def predict_session(payload):

    X = build_features(payload)

    raw_score = float(
        rf_model.predict_proba(X)[0, 1]
    )

    calibrated_probability = float(
        calibrator.predict(
            np.array([raw_score])
        )[0]
    )

    intent = get_intent_segment(
        raw_score
    )

    return {
        "raw_propensity_score":
            round(raw_score, 6),

        "purchase_probability":
            round(calibrated_probability, 6),

        "purchase_probability_pct":
            round(
                calibrated_probability * 100,
                3
            ),

        "intent_segment":
            intent
    }