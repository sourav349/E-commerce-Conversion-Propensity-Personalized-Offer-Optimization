from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


HIGH_INTENT_PAYLOAD = {
    "session_date": "2021-01-15",
    "total_events_5min": 18,
    "page_views_5min": 7,
    "product_views_5min": 5,
    "product_clicks_5min": 2,
    "add_to_cart_5min": 1,
    "begin_checkout_5min": 1,
    "search_events_5min": 0,
    "is_new_user": 0,
    "device_category": "desktop",
    "operating_system": "Macintosh",
    "traffic_source": "google",
    "traffic_medium": "organic",
}


LOW_INTENT_PAYLOAD = {
    "session_date": "2021-01-16",
    "total_events_5min": 3,
    "page_views_5min": 1,
    "product_views_5min": 0,
    "product_clicks_5min": 0,
    "add_to_cart_5min": 0,
    "begin_checkout_5min": 0,
    "search_events_5min": 0,
    "is_new_user": 1,
    "device_category": "desktop",
    "operating_system": "<Other>",
    "traffic_source": "google",
    "traffic_medium": "cpc",
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert "message" in body
    assert body["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "Random Forest"
    assert body["calibration"] == "Isotonic Regression"
    assert body["observation_window"] == "First 5 minutes"
    assert body["prediction_target"] == "Purchase after minute 5"
    assert body["explainability"] == "Live SHAP"


def test_high_intent_prediction():
    response = client.post(
        "/predict",
        json=HIGH_INTENT_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert 0.0 <= body["raw_propensity_score"] <= 1.0
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert 0.0 <= body["purchase_probability_pct"] <= 100.0

    assert body["intent_segment"] in {
        "High Intent",
        "Medium Intent",
        "Low Intent",
    }

    assert body["explanation_available"] is True
    assert isinstance(body["top_positive_drivers"], list)
    assert isinstance(body["top_negative_drivers"], list)

    assert len(body["top_positive_drivers"]) <= 5
    assert len(body["top_negative_drivers"]) <= 5


def test_low_intent_prediction():
    response = client.post(
        "/predict",
        json=LOW_INTENT_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["intent_segment"] == "Low Intent"
    assert body["explanation_available"] is True


def test_high_engagement_scores_above_low_engagement():
    high_response = client.post(
        "/predict",
        json=HIGH_INTENT_PAYLOAD,
    )

    low_response = client.post(
        "/predict",
        json=LOW_INTENT_PAYLOAD,
    )

    assert high_response.status_code == 200
    assert low_response.status_code == 200

    high = high_response.json()
    low = low_response.json()

    assert (
        high["raw_propensity_score"]
        > low["raw_propensity_score"]
    )


def test_negative_event_count_is_rejected():
    bad_payload = {
        **HIGH_INTENT_PAYLOAD,
        "total_events_5min": -1,
    }

    response = client.post(
        "/predict",
        json=bad_payload,
    )

    assert response.status_code == 422


def test_invalid_is_new_user_is_rejected():
    bad_payload = {
        **HIGH_INTENT_PAYLOAD,
        "is_new_user": 2,
    }

    response = client.post(
        "/predict",
        json=bad_payload,
    )

    assert response.status_code == 422


def test_zero_denominator_features_do_not_crash():
    payload = {
        **LOW_INTENT_PAYLOAD,
        "page_views_5min": 0,
        "product_views_5min": 0,
        "product_clicks_5min": 0,
        "add_to_cart_5min": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200


def test_shap_driver_schema():
    response = client.post(
        "/predict",
        json=HIGH_INTENT_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    drivers = (
        body["top_positive_drivers"]
        + body["top_negative_drivers"]
    )

    assert drivers

    for driver in drivers:
        assert set(driver) == {
            "feature",
            "value",
            "shap_value",
            "direction",
        }

        assert driver["direction"] in {
            "up",
            "down",
        }
