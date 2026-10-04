import json
import sys
import requests

API_URL = "http://127.0.0.1:8001"

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


def fail(message):
    print("\n❌ " + message)
    sys.exit(1)


def pretty(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
    except requests.RequestException as exc:
        fail(
            "Could not connect to FastAPI.\n"
            "Start it first with:\n"
            "python3 -m uvicorn backend.main:app --reload --port 8001\n\n"
            f"Original error: {exc}"
        )

    if response.status_code != 200:
        fail(
            f"/health returned HTTP {response.status_code}\n"
            f"{response.text}"
        )

    print("✅ Backend health check passed")


def predict(name, payload):
    print("\n" + "=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        fail(
            f"{name}: /predict returned HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()
    print(pretty(result))

    required_keys = {
        "raw_propensity_score",
        "purchase_probability",
        "purchase_probability_pct",
        "intent_segment",
        "explanation_available",
        "explanation_note",
        "top_positive_drivers",
        "top_negative_drivers",
    }

    missing = required_keys - result.keys()
    if missing:
        fail(
            f"{name}: response is missing keys: {sorted(missing)}"
        )

    if result["intent_segment"] not in {
        "High Intent",
        "Medium Intent",
        "Low Intent",
    }:
        fail(
            f"{name}: invalid intent segment: "
            f"{result['intent_segment']}"
        )

    if not isinstance(result["top_positive_drivers"], list):
        fail(f"{name}: top_positive_drivers must be a list")

    if not isinstance(result["top_negative_drivers"], list):
        fail(f"{name}: top_negative_drivers must be a list")

    if result["explanation_available"] is not True:
        fail(
            f"{name}: SHAP explanation is not available.\n"
            f"Note: {result.get('explanation_note')}"
        )

    print(f"✅ {name}: prediction response is valid")
    print(f"✅ {name}: live SHAP is available")

    return result


def print_driver_summary(name, result):
    print(f"\n{name} summary")
    print("-" * 70)
    print(f"Probability: {result['purchase_probability_pct']:.2f}%")
    print(f"Raw score:   {result['raw_propensity_score']:.6f}")
    print(f"Segment:     {result['intent_segment']}")

    print("\nTop positive SHAP drivers:")
    if result["top_positive_drivers"]:
        for driver in result["top_positive_drivers"]:
            print(
                f"  ↑ {driver['feature']:<30} "
                f"value={str(driver.get('value', '')):<10} "
                f"SHAP={driver['shap_value']:+.6f}"
            )
    else:
        print("  None returned")

    print("\nTop negative SHAP drivers:")
    if result["top_negative_drivers"]:
        for driver in result["top_negative_drivers"]:
            print(
                f"  ↓ {driver['feature']:<30} "
                f"value={str(driver.get('value', '')):<10} "
                f"SHAP={driver['shap_value']:+.6f}"
            )
    else:
        print("  None returned")


def main():
    print("LIVE SHAP END-TO-END VERIFICATION")
    print("=" * 70)

    check_health()

    high = predict(
        "High-engagement session",
        HIGH_INTENT_PAYLOAD,
    )

    low = predict(
        "Low-engagement session",
        LOW_INTENT_PAYLOAD,
    )

    print_driver_summary(
        "HIGH-ENGAGEMENT SESSION",
        high,
    )

    print_driver_summary(
        "LOW-ENGAGEMENT SESSION",
        low,
    )

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        "High-engagement raw score: "
        f"{high['raw_propensity_score']:.6f}"
    )
    print(
        "Low-engagement raw score:  "
        f"{low['raw_propensity_score']:.6f}"
    )

    if high["raw_propensity_score"] > low["raw_propensity_score"]:
        print(
            "✅ High-engagement session has a higher "
            "raw propensity score."
        )
    else:
        print(
            "⚠️ High-engagement session did not receive a higher "
            "raw score. Inspect the feature engineering/model behavior."
        )

    print(
        "\n🎉 Backend prediction + calibration + segmentation "
        "+ live SHAP verification completed."
    )
    print(
        "\nNext: run Streamlit and confirm the same SHAP drivers "
        "are displayed in the UI."
    )


if __name__ == "__main__":
    main()
