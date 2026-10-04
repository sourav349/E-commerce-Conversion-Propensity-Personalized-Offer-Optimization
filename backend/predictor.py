from pathlib import Path
from collections import defaultdict
import json
import logging

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
CALIBRATOR_PATH = BASE_DIR / "models" / "isotonic_calibrator.pkl"
CONFIG_PATH = BASE_DIR / "models" / "deployment_config.json"


# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------

rf_pipeline = joblib.load(MODEL_PATH)

isotonic_calibrator = joblib.load(
    CALIBRATOR_PATH
)

with open(CONFIG_PATH, "r") as f:
    deployment_config = json.load(f)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

NUMERIC_FEATURES = [
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
]


CATEGORICAL_FEATURES = [
    "device_category",
    "operating_system",
    "traffic_source",
    "traffic_medium",
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# DEPLOYMENT THRESHOLDS
# ---------------------------------------------------------

HIGH_THRESHOLD = float(
    deployment_config["high_intent_raw_score_threshold"]
)

MEDIUM_THRESHOLD = float(
    deployment_config["medium_intent_raw_score_threshold"]
)


# ---------------------------------------------------------
# GET MODEL COMPONENTS FOR SHAP
# ---------------------------------------------------------

# The saved model should be:
#
# Pipeline(
#     preprocessing
#     -> RandomForestClassifier
# )

if hasattr(rf_pipeline, "steps"):

    # Everything except final Random Forest
    transformer_pipeline = rf_pipeline[:-1]

    # Final classifier
    rf_classifier = rf_pipeline.steps[-1][1]

else:
    transformer_pipeline = None
    rf_classifier = rf_pipeline


# ---------------------------------------------------------
# INITIALIZE SHAP ONCE
# ---------------------------------------------------------

SHAP_EXPLAINER = None

if shap is not None:
    try:
        SHAP_EXPLAINER = shap.TreeExplainer(
            rf_classifier
        )
    except Exception:
        logger.exception(
            "Could not initialize SHAP TreeExplainer."
        )


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def safe_ratio(
    numerator,
    denominator
):
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def value_to_string(value):

    if isinstance(
        value,
        (np.integer,)
    ):
        return str(int(value))

    if isinstance(
        value,
        (np.floating,)
    ):
        return f"{float(value):.4f}"

    return str(value)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def build_features(session):

    # Convert session date
    session_date = pd.Timestamp(
        session.session_date
    )

    day_of_week = (
        session_date.dayofweek
    )

    is_weekend = int(
        day_of_week in [5, 6]
    )

    # -----------------------------------------------------
    # Derived behavior flags
    # -----------------------------------------------------

    has_cart = int(
        session.add_to_cart_5min > 0
    )

    has_checkout = int(
        session.begin_checkout_5min > 0
    )

    has_product_view = int(
        session.product_views_5min > 0
    )

    has_product_click = int(
        session.product_clicks_5min > 0
    )

    is_returning_user = (
        1 - session.is_new_user
    )

    # -----------------------------------------------------
    # Ratio features
    # -----------------------------------------------------

    product_views_per_page = (
        safe_ratio(
            session.product_views_5min,
            session.page_views_5min
        )
    )

    click_per_product_view = (
        safe_ratio(
            session.product_clicks_5min,
            session.product_views_5min
        )
    )

    cart_per_product_view = (
        safe_ratio(
            session.add_to_cart_5min,
            session.product_views_5min
        )
    )

    # -----------------------------------------------------
    # MODEL ROW
    # -----------------------------------------------------

    row = {
        "total_events_5min":
            session.total_events_5min,

        "page_views_5min":
            session.page_views_5min,

        "product_views_5min":
            session.product_views_5min,

        "product_clicks_5min":
            session.product_clicks_5min,

        "add_to_cart_5min":
            session.add_to_cart_5min,

        "begin_checkout_5min":
            session.begin_checkout_5min,

        "search_events_5min":
            session.search_events_5min,

        "is_returning_user":
            is_returning_user,

        "has_cart":
            has_cart,

        "has_checkout":
            has_checkout,

        "has_product_view":
            has_product_view,

        "has_product_click":
            has_product_click,

        "product_views_per_page":
            product_views_per_page,

        "click_per_product_view":
            click_per_product_view,

        "cart_per_product_view":
            cart_per_product_view,

        "is_weekend":
            is_weekend,

        "device_category":
            session.device_category,

        "operating_system":
            session.operating_system,

        "traffic_source":
            session.traffic_source,

        "traffic_medium":
            session.traffic_medium,
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )


# ---------------------------------------------------------
# INTENT SEGMENT
# ---------------------------------------------------------

def get_intent_segment(
    raw_probability
):

    if raw_probability >= HIGH_THRESHOLD:
        return "High Intent"

    if raw_probability >= MEDIUM_THRESHOLD:
        return "Medium Intent"

    return "Low Intent"


# ---------------------------------------------------------
# TRANSFORMED FEATURE -> ORIGINAL FEATURE
# ---------------------------------------------------------

def get_source_feature(
    transformed_name
):

    # Example:
    #
    # num__add_to_cart_5min
    #
    # becomes:
    #
    # add_to_cart_5min

    clean_name = (
        transformed_name
        .split("__", 1)[-1]
    )

    # Numeric
    if clean_name in NUMERIC_FEATURES:
        return clean_name

    # Categorical
    #
    # Example:
    #
    # device_category_mobile
    #
    # maps to:
    #
    # device_category

    for feature in CATEGORICAL_FEATURES:

        if (
            clean_name == feature
            or clean_name.startswith(
                feature + "_"
            )
        ):
            return feature

    return clean_name


# ---------------------------------------------------------
# GET POSITIVE-CLASS SHAP VALUES
# ---------------------------------------------------------

def extract_positive_class_shap(
    shap_values
):

    # -----------------------------------------------------
    # Older SHAP versions:
    #
    # [
    #   class_0_values,
    #   class_1_values
    # ]
    # -----------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            return np.asarray(
                shap_values[1]
            )[0]

        return np.asarray(
            shap_values[0]
        )[0]


    values = np.asarray(
        shap_values
    )

    # -----------------------------------------------------
    # Newer SHAP versions:
    #
    # shape:
    #
    # (samples, features, classes)
    # -----------------------------------------------------

    if values.ndim == 3:

        if values.shape[-1] >= 2:

            return values[
                0,
                :,
                1
            ]

        return values[
            0,
            :,
            0
        ]


    # Standard:
    #
    # (samples, features)

    if values.ndim == 2:
        return values[0]


    if values.ndim == 1:
        return values


    raise ValueError(
        f"Unexpected SHAP shape: "
        f"{values.shape}"
    )


# ---------------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------------

def explain_prediction(
    model_input,
    top_n=5
):

    if shap is None:

        return (
            [],
            [],
            False,
            "SHAP package is not installed."
        )

    if SHAP_EXPLAINER is None:

        return (
            [],
            [],
            False,
            "SHAP explainer could not be initialized."
        )


    # -----------------------------------------------------
    # Apply exactly the same preprocessing used by RF
    # -----------------------------------------------------

    if transformer_pipeline is not None:

        transformed = (
            transformer_pipeline
            .transform(
                model_input
            )
        )

        try:
            transformed_names = (
                transformer_pipeline
                .get_feature_names_out()
            )

        except Exception:

            transformed_names = [
                f"feature_{i}"
                for i in range(
                    transformed.shape[1]
                )
            ]

    else:

        transformed = (
            model_input.values
        )

        transformed_names = (
            model_input.columns
        )


    # SHAP works more reliably with dense
    # matrix for this small single-row case

    if hasattr(
        transformed,
        "toarray"
    ):
        transformed_dense = (
            transformed.toarray()
        )
    else:
        transformed_dense = (
            np.asarray(transformed)
        )


    # -----------------------------------------------------
    # CALCULATE SHAP
    # -----------------------------------------------------

    try:

        shap_values = (
            SHAP_EXPLAINER
            .shap_values(
                transformed_dense,
                check_additivity=False
            )
        )

    except TypeError:

        shap_values = (
            SHAP_EXPLAINER
            .shap_values(
                transformed_dense
            )
        )


    positive_class_values = (
        extract_positive_class_shap(
            shap_values
        )
    )


    # -----------------------------------------------------
    # Aggregate one-hot encoded categories back to
    # original source features.
    #
    # Example:
    #
    # cat__device_category_mobile
    # cat__device_category_desktop
    #
    # becomes:
    #
    # device_category
    # -----------------------------------------------------

    aggregated = defaultdict(float)

    for (
        transformed_feature,
        shap_value
    ) in zip(
        transformed_names,
        positive_class_values
    ):

        source_feature = (
            get_source_feature(
                str(
                    transformed_feature
                )
            )
        )

        aggregated[source_feature] += (
            float(shap_value)
        )


    # -----------------------------------------------------
    # POSITIVE DRIVERS
    # -----------------------------------------------------

    positive = sorted(
        [
            (feature, value)
            for feature, value
            in aggregated.items()
            if value > 0
        ],
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


    # -----------------------------------------------------
    # NEGATIVE DRIVERS
    # -----------------------------------------------------

    negative = sorted(
        [
            (feature, value)
            for feature, value
            in aggregated.items()
            if value < 0
        ],
        key=lambda x: x[1]
    )[:top_n]


    # -----------------------------------------------------
    # BUILD RESPONSE
    # -----------------------------------------------------

    positive_drivers = []

    for feature, shap_value in positive:

        original_value = (
            model_input.iloc[0]
            .get(
                feature,
                ""
            )
        )

        positive_drivers.append(
            {
                "feature":
                    feature,

                "value":
                    value_to_string(
                        original_value
                    ),

                "shap_value":
                    round(
                        float(shap_value),
                        6
                    ),

                "direction":
                    "up",
            }
        )


    negative_drivers = []

    for feature, shap_value in negative:

        original_value = (
            model_input.iloc[0]
            .get(
                feature,
                ""
            )
        )

        negative_drivers.append(
            {
                "feature":
                    feature,

                "value":
                    value_to_string(
                        original_value
                    ),

                "shap_value":
                    round(
                        float(shap_value),
                        6
                    ),

                "direction":
                    "down",
            }
        )


    return (
        positive_drivers,
        negative_drivers,
        True,
        (
            "SHAP explains the underlying "
            "Random Forest propensity score. "
            "Isotonic calibration is applied "
            "after the Random Forest."
        )
    )


# ---------------------------------------------------------
# MAIN PREDICTION
# ---------------------------------------------------------

def predict_session(
    session
):

    model_input = (
        build_features(
            session
        )
    )


    # -----------------------------------------------------
    # RAW RANDOM FOREST PROPENSITY
    # -----------------------------------------------------

    raw_probability = float(
        rf_pipeline
        .predict_proba(
            model_input
        )[0, 1]
    )


    # -----------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------

    calibrated_probability = float(
        isotonic_calibrator
        .predict(
            [raw_probability]
        )[0]
    )


    calibrated_probability = float(
        np.clip(
            calibrated_probability,
            0.0,
            1.0
        )
    )


    # -----------------------------------------------------
    # SEGMENT
    # -----------------------------------------------------

    intent_segment = (
        get_intent_segment(
            raw_probability
        )
    )


    # -----------------------------------------------------
    # LIVE SHAP
    # -----------------------------------------------------

    try:

        (
            positive_drivers,
            negative_drivers,
            explanation_available,
            explanation_note
        ) = explain_prediction(
            model_input=model_input,
            top_n=5
        )

    except Exception:

        logger.exception(
            "Live SHAP explanation failed."
        )

        positive_drivers = []
        negative_drivers = []

        explanation_available = False

        explanation_note = (
            "Prediction succeeded, but "
            "SHAP explanation was unavailable."
        )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "raw_propensity_score":
            round(
                raw_probability,
                6
            ),

        "purchase_probability":
            round(
                calibrated_probability,
                6
            ),

        "purchase_probability_pct":
            round(
                calibrated_probability
                * 100,
                2
            ),

        "intent_segment":
            intent_segment,

        "explanation_available":
            explanation_available,

        "explanation_note":
            explanation_note,

        "top_positive_drivers":
            positive_drivers,

        "top_negative_drivers":
            negative_drivers,
    }