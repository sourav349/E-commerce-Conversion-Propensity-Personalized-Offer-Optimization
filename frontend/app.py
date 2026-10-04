import os

from datetime import date



import pandas as pd

import requests

import streamlit as st





# =========================================================

# CONFIGURATION

# =========================================================



API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")



st.set_page_config(

    page_title="E-commerce Conversion Intelligence",

    page_icon="🛒",

    layout="wide",

)





# =========================================================

# HELPERS

# =========================================================



def friendly_feature_name(feature: str) -> str:

    """Convert backend feature names into business-friendly labels."""

    names = {

        "total_events_5min": "Total session activity",

        "page_views_5min": "Page views",

        "product_views_5min": "Product views",

        "product_clicks_5min": "Product clicks",

        "add_to_cart_5min": "Add-to-cart activity",

        "begin_checkout_5min": "Checkout activity",

        "search_events_5min": "Search activity",

        "is_returning_user": "Returning user",

        "has_cart": "Has cart activity",

        "has_checkout": "Started checkout",

        "has_product_view": "Viewed a product",

        "has_product_click": "Clicked a product",

        "product_views_per_page": "Product views per page",

        "click_per_product_view": "Clicks per product view",

        "cart_per_product_view": "Cart actions per product view",

        "is_weekend": "Weekend session",

        "device_category": "Device",

        "operating_system": "Operating system",

        "traffic_source": "Traffic source",

        "traffic_medium": "Traffic medium",

    }

    return names.get(feature, feature.replace("_", " ").title())





@st.cache_data(ttl=60, show_spinner=False)
def get_backend_status() -> bool:

    """Check whether the FastAPI backend is reachable.

    Render free services can sleep after inactivity, so the first health
    request may take up to about a minute while the backend wakes up.
    """

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=65,
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:

        return False





def render_shap_explanation(result: dict) -> None:

    """Render live/local SHAP drivers returned by the API."""

    st.subheader("Why did the model predict this?")



    if not result.get("explanation_available", False):

        st.warning(

            result.get(

                "explanation_note",

                "Live SHAP explanation is currently unavailable.",

            )

        )

        return



    positive_drivers = result.get("top_positive_drivers", [])

    negative_drivers = result.get("top_negative_drivers", [])



    left, right = st.columns(2)



    with left:

        st.markdown("### ↑ Increasing Purchase Propensity")

        if positive_drivers:

            for driver in positive_drivers:

                feature_name = friendly_feature_name(driver.get("feature", ""))

                input_value = driver.get("value", "")

                shap_value = float(driver.get("shap_value", 0.0))

                st.success(

                    f"**{feature_name}**\n\n"

                    f"Input value: `{input_value}`\n\n"

                    f"SHAP contribution: `{shap_value:+.4f}`"

                )

        else:

            st.info("No positive SHAP drivers were returned.")



    with right:

        st.markdown("### ↓ Decreasing Purchase Propensity")

        if negative_drivers:

            for driver in negative_drivers:

                feature_name = friendly_feature_name(driver.get("feature", ""))

                input_value = driver.get("value", "")

                shap_value = float(driver.get("shap_value", 0.0))

                st.warning(

                    f"**{feature_name}**\n\n"

                    f"Input value: `{input_value}`\n\n"

                    f"SHAP contribution: `{shap_value:+.4f}`"

                )

        else:

            st.info("No negative SHAP drivers were returned.")



    all_drivers = []

    for driver in positive_drivers + negative_drivers:

        all_drivers.append(

            {

                "Feature": friendly_feature_name(driver.get("feature", "")),

                "SHAP Contribution": float(driver.get("shap_value", 0.0)),

            }

        )



    if all_drivers:

        shap_df = pd.DataFrame(all_drivers)

        shap_df["Absolute Contribution"] = shap_df["SHAP Contribution"].abs()

        shap_df = (

            shap_df.sort_values("Absolute Contribution", ascending=False)

            .drop(columns="Absolute Contribution")

        )



        st.markdown("### Local SHAP Contributions")

        st.bar_chart(

            shap_df.set_index("Feature")[["SHAP Contribution"]],

            use_container_width=True,

        )



    note = result.get("explanation_note", "")

    if note:

        st.caption(note)



    st.caption(

        "SHAP explains the underlying Random Forest propensity score. "

        "Isotonic calibration is applied afterward to obtain the final probability."

    )





def render_business_strategy(intent: str) -> None:

    st.subheader("Recommended Business Strategy")



    if intent == "High Intent":

        st.success(

            """

**High purchase intent detected**



- Avoid unnecessary blanket discounts

- Focus on checkout completion

- Recommend relevant products

- Use cart / checkout reminders

- Preserve margin where possible

"""

        )

    elif intent == "Medium Intent":

        st.warning(

            """

**Medium purchase intent detected**



- Candidate for controlled campaigns

- Test incentives using randomized A/B experiments

- Measure incremental uplift

- Use campaign economics before scaling

- Do not assume that a discount automatically causes conversion

"""

        )

    else:

        st.info(

            """

**Low immediate purchase intent**



- Focus on product discovery

- Consider remarketing / retargeting

- Avoid automatically providing discounts

- Test whether promotions actually change behaviour

"""

        )





# =========================================================

# HEADER

# =========================================================



st.title("🛒 E-commerce Conversion Intelligence")

st.caption(

    "Predict purchase propensity from the first 5 minutes of customer-session "

    "behaviour and explain each prediction with live SHAP."

)





# =========================================================

# BACKEND STATUS

# =========================================================



backend_online = get_backend_status()



if backend_online:

    st.success(f"FastAPI backend connected: {API_URL}", icon="✅")

else:

    st.info(

        "Backend may be waking up from Render free-tier sleep. "

        "The first request can take around 30–60 seconds. "

        "You can still click Predict Conversion; the prediction request will "

        "also try to wake the backend.",

        icon="⏳",

    )





# =========================================================

# TABS

# =========================================================



tab1, tab2, tab3, tab4 = st.tabs(

    [

        "🧠 Live Prediction",

        "📊 Model Performance",

        "👥 Intent Segments",

        "💰 Campaign Simulator",

    ]

)





# =========================================================

# TAB 1 — LIVE PREDICTION

# =========================================================



with tab1:

    st.header("Live Purchase Propensity Prediction")

    st.write(

        "Enter behaviour observed during the first 5 minutes of a customer "

        "session. Every time you click **Predict Conversion**, the backend "

        "calculates a new propensity score, calibrated probability, intent "

        "segment, and SHAP explanation for that exact session."

    )



    st.divider()

    col1, col2, col3 = st.columns(3)



    with col1:

        st.subheader("Session Activity")

        session_date = st.date_input(

            "Session Date",

            value=date(2021, 1, 15),

            help="The original model was trained using 2020–2021 session data.",

        )

        total_events = st.number_input(

            "Total Events (first 5 min)", min_value=0, value=10, step=1

        )

        page_views = st.number_input(

            "Page Views", min_value=0, value=4, step=1

        )

        product_views = st.number_input(

            "Product Views", min_value=0, value=3, step=1

        )



    with col2:

        st.subheader("Purchase Behaviour")

        product_clicks = st.number_input(

            "Product Clicks", min_value=0, value=1, step=1

        )

        add_to_cart = st.number_input(

            "Add to Cart Events", min_value=0, value=1, step=1

        )

        begin_checkout = st.number_input(

            "Checkout Events",

            min_value=0,

            value=0,

            step=1,

            help=(

                "Checkout means the customer started the process of completing "

                "the purchase. It does not mean the purchase was completed."

            ),

        )

        search_events = st.number_input(

            "Search Events", min_value=0, value=0, step=1

        )



    with col3:

        st.subheader("Customer Context")

        user_type = st.selectbox(

            "User Type", ["Returning User", "New User"]

        )

        device_category = st.selectbox(

            "Device", ["desktop", "mobile", "tablet"]

        )

        operating_system = st.selectbox(

            "Operating System",

            ["Macintosh", "Windows", "Android", "iOS", "Linux", "<Other>"],

        )

        traffic_source = st.text_input("Traffic Source", value="google")

        traffic_medium = st.text_input("Traffic Medium", value="organic")



    st.divider()



    predict_clicked = st.button(

        "Predict Conversion",

        type="primary",

        use_container_width=True,

    )



    if predict_clicked:

        if not backend_online:

            st.info(

                "Backend may be waking up. The prediction request will wait for it.",

                icon="⏳",

            )

        payload = {

            "session_date": str(session_date),

            "total_events_5min": int(total_events),

            "page_views_5min": int(page_views),

            "product_views_5min": int(product_views),

            "product_clicks_5min": int(product_clicks),

            "add_to_cart_5min": int(add_to_cart),

            "begin_checkout_5min": int(begin_checkout),

            "search_events_5min": int(search_events),

            "is_new_user": 1 if user_type == "New User" else 0,

            "device_category": device_category,

            "operating_system": operating_system,

            "traffic_source": traffic_source.strip() or "<Other>",

            "traffic_medium": traffic_medium.strip() or "<Other>",

        }



        try:

            with st.spinner(

                "Scoring session and calculating live SHAP explanation..."

            ):

                response = requests.post(

                    f"{API_URL}/predict",

                    json=payload,

                    timeout=60,

                )



            if response.status_code == 200:

                result = response.json()

                probability = float(result["purchase_probability_pct"])

                intent = result["intent_segment"]

                raw_score = float(result["raw_propensity_score"])



                st.subheader("Prediction Result")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric("Purchase Probability", f"{probability:.2f}%")

                metric2.metric("Intent Segment", intent)

                metric3.metric("Raw Propensity Score", f"{raw_score:.4f}")



                st.progress(min(max(probability / 100.0, 0.0), 1.0))



                st.subheader("How to Interpret This")

                st.markdown(

                    f"""

The model estimates approximately **{probability:.2f}% purchase probability**

for a purchase occurring after the initial 5-minute observation window.



The session is classified as **{intent}** using the underlying Random Forest

propensity score and validation-derived segment thresholds.

"""

                )



                st.divider()

                render_shap_explanation(result)



                st.divider()

                render_business_strategy(intent)



                with st.expander("View API Request / Response"):

                    st.markdown("### Request")

                    st.json(payload)

                    st.markdown("### Response")

                    st.json(result)



            else:

                st.error(f"Backend returned HTTP {response.status_code}.")

                st.code(response.text)



        except requests.exceptions.Timeout:

            st.error(

                "The prediction request timed out. SHAP can take longer than "

                "a normal prediction, so check the backend logs and try again."

            )

        except requests.exceptions.RequestException as exc:

            st.error("Could not connect to FastAPI.")

            st.exception(exc)





# =========================================================

# TAB 2 — MODEL PERFORMANCE

# =========================================================



with tab2:

    st.header("Model Performance")

    st.write("Final performance measured on the held-out future test period.")

    st.divider()



    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("ROC-AUC", "0.952")

    metric2.metric("PR-AUC", "0.258")

    metric3.metric("Lift @ Top 5%", "15.51×")

    metric4.metric("Recall @ Top 5%", "77.54%")



    metric5, metric6, metric7 = st.columns(3)

    metric5.metric("Log Loss", "0.0314")

    metric6.metric("Brier Score", "0.0076")

    metric7.metric("Test Conversion Rate", "0.93%")



    st.divider()

    st.subheader("Targeting Performance")



    lift_data = pd.DataFrame(

        {

            "Targeted Population": [

                "Top 1%", "Top 5%", "Top 10%", "Top 20%", "Top 30%"

            ],

            "Conversion Rate (%)": [

                33.514690, 14.425587, 8.202785, 4.340495, 2.937230

            ],

            "Recall (%)": [

                36.023392, 77.543860, 88.187135, 93.333333, 94.736842

            ],

            "Lift": [

                36.033583, 15.509784, 8.819289, 4.666717, 3.157986

            ],

        }

    )



    st.dataframe(lift_data.round(3), use_container_width=True, hide_index=True)



    st.subheader("Lift by Targeting Depth")

    st.bar_chart(

        lift_data[["Targeted Population", "Lift"]].set_index("Targeted Population"),

        use_container_width=True,

    )



    st.info(

        """

Targeting only the **top 5%** highest-scoring sessions captured approximately

**77.5% of purchasers** and achieved about **15.5× lift** over the overall test

conversion rate.

"""

    )



    st.subheader("What the Metrics Mean")

    st.markdown(

        """

**ROC-AUC:** Measures how well the model ranks purchasers above non-purchasers

across thresholds.



**PR-AUC:** Precision-recall performance, especially useful because purchases are rare.



**Lift:** How much higher conversion is in a selected high-scoring population

compared with the overall population.



**Brier Score:** Measures probability accuracy. Lower is better.



**Log Loss:** Penalizes poor probability estimates, especially confident mistakes.

"""

    )





# =========================================================

# TAB 3 — INTENT SEGMENTS

# =========================================================



with tab3:

    st.header("Customer Intent Segmentation")

    st.write("Sessions were ranked using the Random Forest propensity score.")



    segment_df = pd.DataFrame(

        {

            "Intent Segment": ["High Intent", "Medium Intent", "Low Intent"],

            "Sessions": [4596, 13789, 73541],

            "Purchases": [664, 134, 57],

            "Session Share (%)": [4.999674, 15.000109, 80.000218],

            "Purchase Share (%)": [77.660819, 15.672515, 6.666667],

            "Actual Conversion Rate (%)": [14.447346, 0.971789, 0.077508],

            "Predicted Conversion Rate (%)": [12.928111, 0.986399, 0.089116],

        }

    )



    st.dataframe(segment_df.round(3), use_container_width=True, hide_index=True)

    st.divider()



    st.subheader("Conversion Rate by Intent Segment")

    st.bar_chart(

        segment_df[["Intent Segment", "Actual Conversion Rate (%)"]].set_index(

            "Intent Segment"

        ),

        use_container_width=True,

    )



    st.subheader("Purchase Concentration")

    st.bar_chart(

        segment_df[["Intent Segment", "Purchase Share (%)"]].set_index(

            "Intent Segment"

        ),

        use_container_width=True,

    )



    st.success(

        """

**Key finding**



High Intent represents only about **5% of sessions** but contains approximately

**77.7% of all purchases**.



The High-Intent group converted at approximately **14.45%**, compared with only

**0.078%** among Low-Intent sessions.

"""

    )



    st.subheader("Business Interpretation")

    high_col, medium_col, low_col = st.columns(3)



    with high_col:

        st.markdown(

            """

### 🔥 High Intent



**5% of sessions**  

**77.7% of purchases**  

**14.45% conversion**



These sessions already demonstrate strong purchase behaviour.



**Strategy**



Avoid unnecessary blanket discounts. Focus on checkout completion,

recommendations, and customer experience.

"""

        )



    with medium_col:

        st.markdown(

            """

### ⚡ Medium Intent



**15% of sessions**  

**15.7% of purchases**  

**0.97% conversion**



Some purchase intent exists, but conversion remains relatively low.



**Strategy**



Use controlled randomized experiments to test whether offers create incremental

conversions.

"""

        )



    with low_col:

        st.markdown(

            """

### ❄️ Low Intent



**80% of sessions**  

**6.7% of purchases**  

**0.078% conversion**



Immediate purchase intent is very low.



**Strategy**



Focus on product discovery, remarketing, and engagement instead of automatically

giving discounts.

"""

        )





# =========================================================

# TAB 4 — CAMPAIGN SIMULATOR

# =========================================================



with tab4:

    st.header("Campaign Profit Simulator")

    st.write(

        """

Estimate whether a discount campaign could become profitable for the Medium-Intent

segment.



These values are **business assumptions**, not values learned from the GA4 dataset.

"""

    )



    st.divider()

    input1, input2, input3 = st.columns(3)



    with input1:

        gross_margin = st.number_input(

            "Gross Margin per Order (₹)",

            min_value=0.0,

            value=600.0,

            step=50.0,

        )



    with input2:

        discount = st.number_input(

            "Discount per Converted Order (₹)",

            min_value=0.0,

            value=100.0,

            step=10.0,

        )



    with input3:

        campaign_cost = st.number_input(

            "Campaign Cost per Session (₹)",

            min_value=0.0,

            value=2.0,

            step=0.5,

        )



    uplift_pp = st.slider(

        "Expected Incremental Conversion Uplift (percentage points)",

        min_value=0.0,

        max_value=3.0,

        value=0.50,

        step=0.05,

    )



    medium_sessions = 13789

    baseline_conversion_rate = 0.971789 / 100

    baseline_orders = medium_sessions * baseline_conversion_rate



    new_conversion_rate = min(

        baseline_conversion_rate + uplift_pp / 100,

        1.0,

    )



    campaign_orders = medium_sessions * new_conversion_rate

    baseline_profit = baseline_orders * gross_margin

    campaign_profit = (

        campaign_orders * (gross_margin - discount)

        - medium_sessions * campaign_cost

    )

    incremental_profit = campaign_profit - baseline_profit

    additional_orders = campaign_orders - baseline_orders



    result1, result2, result3, result4 = st.columns(4)

    result1.metric("Baseline Orders", f"{baseline_orders:,.1f}")

    result2.metric(

        "Expected Campaign Orders",

        f"{campaign_orders:,.1f}",

        delta=f"{additional_orders:,.1f}",

    )

    result3.metric("Baseline Profit", f"₹{baseline_profit:,.0f}")

    result4.metric("Incremental Profit", f"₹{incremental_profit:,.0f}")



    st.divider()

    st.subheader("Campaign Economics")

    left, right = st.columns(2)



    with left:

        st.metric(

            "Baseline Conversion",

            f"{baseline_conversion_rate * 100:.3f}%",

        )



    with right:

        st.metric(

            "Simulated Campaign Conversion",

            f"{new_conversion_rate * 100:.3f}%",

            delta=f"+{uplift_pp:.2f} pp",

        )



    if gross_margin > discount:

        break_even_uplift = (

            discount * baseline_conversion_rate + campaign_cost

        ) / (gross_margin - discount)



        break_even_uplift_pp = break_even_uplift * 100

        break_even_conversion = baseline_conversion_rate + break_even_uplift

        break_even_additional_orders = medium_sessions * break_even_uplift



        st.subheader("Break-even Analysis")

        b1, b2, b3 = st.columns(3)

        b1.metric("Required Break-even Uplift", f"{break_even_uplift_pp:.3f} pp")

        b2.metric(

            "Required Conversion Rate",

            f"{break_even_conversion * 100:.3f}%",

        )

        b3.metric(

            "Approx. Additional Orders Needed",

            f"{break_even_additional_orders:,.1f}",

        )



        if uplift_pp > break_even_uplift_pp:

            st.success(

                f"Under these assumptions, the simulated campaign is **profitable**.\n\n"

                f"Expected incremental profit: **₹{incremental_profit:,.0f}**"

            )

        elif abs(uplift_pp - break_even_uplift_pp) < 0.01:

            st.warning("The campaign is approximately at break-even.")

        else:

            st.error(

                f"Under these assumptions, the campaign does **not** break even.\n\n"

                f"Expected incremental profit: **₹{incremental_profit:,.0f}**"

            )

    else:

        st.error(

            "Discount must be smaller than gross margin per order for this "

            "break-even model to be meaningful."

        )



    st.divider()

    st.info(

        """

Important: this simulator does **not prove** that a discount will cause the assumed

uplift.



Real incremental uplift should be measured using randomized A/B testing and may

later be modeled using uplift / causal ML techniques.

"""

    )





# =========================================================

# FOOTER

# =========================================================



st.divider()

st.caption(

    "Model: Random Forest | "

    "Calibration: Isotonic Regression | "

    "Explainability: Live SHAP | "

    "Observation Window: First 5 Minutes"

)
