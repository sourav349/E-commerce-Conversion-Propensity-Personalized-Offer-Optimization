import requests
import pandas as pd
import streamlit as st

from datetime import date


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8001"


st.set_page_config(
    page_title="E-commerce Conversion Intelligence",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🛒 E-commerce Conversion Intelligence")

st.caption(
    "Predict purchase propensity from the first 5 minutes "
    "of customer session behaviour."
)


# =========================================================
# BACKEND STATUS
# =========================================================

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3
    )

    backend_online = (
        health_response.status_code == 200
    )

except requests.exceptions.RequestException:
    backend_online = False


if backend_online:
    st.success(
        "FastAPI backend connected.",
        icon="✅"
    )
else:
    st.error(
        "FastAPI backend is not available on port 8001. "
        "Start the backend before using Live Prediction.",
        icon="⚠️"
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧠 Live Prediction",
        "📊 Model Performance",
        "👥 Intent Segments",
        "💰 Campaign Simulator"
    ]
)


# =========================================================
# TAB 1 — LIVE PREDICTION
# =========================================================

with tab1:

    st.header("Live Purchase Propensity Prediction")

    st.write(
        "Enter behaviour observed during the first "
        "5 minutes of a customer session."
    )

    st.divider()

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # LEFT COLUMN
    with col1:

        st.subheader("Session Activity")

        session_date = st.date_input(
            "Session Date",
            value=date(2021, 1, 15),
            help=(
                "The original model was trained using "
                "2020–2021 session data."
            )
        )

        total_events = st.number_input(
            "Total Events (first 5 min)",
            min_value=0,
            value=10,
            step=1
        )

        page_views = st.number_input(
            "Page Views",
            min_value=0,
            value=4,
            step=1
        )

        product_views = st.number_input(
            "Product Views",
            min_value=0,
            value=3,
            step=1
        )


    # MIDDLE COLUMN
    with col2:

        st.subheader("Purchase Behaviour")

        product_clicks = st.number_input(
            "Product Clicks",
            min_value=0,
            value=1,
            step=1
        )

        add_to_cart = st.number_input(
            "Add to Cart Events",
            min_value=0,
            value=1,
            step=1
        )

        begin_checkout = st.number_input(
            "Checkout Events",
            min_value=0,
            value=0,
            step=1
        )

        search_events = st.number_input(
            "Search Events",
            min_value=0,
            value=0,
            step=1
        )


    # RIGHT COLUMN
    with col3:

        st.subheader("Customer Context")

        user_type = st.selectbox(
            "User Type",
            [
                "Returning User",
                "New User"
            ]
        )

        device_category = st.selectbox(
            "Device",
            [
                "desktop",
                "mobile",
                "tablet"
            ]
        )

        operating_system = st.selectbox(
            "Operating System",
            [
                "Macintosh",
                "Windows",
                "Android",
                "iOS",
                "Linux",
                "<Other>"
            ]
        )

        traffic_source = st.text_input(
            "Traffic Source",
            value="google"
        )

        traffic_medium = st.text_input(
            "Traffic Medium",
            value="organic"
        )


    st.divider()


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    predict_clicked = st.button(
        "Predict Conversion",
        type="primary",
        use_container_width=True
    )


    if predict_clicked:

        if not backend_online:

            st.error(
                "Prediction cannot run because "
                "the FastAPI backend is offline."
            )

        else:

            payload = {

                "session_date":
                    str(session_date),

                "total_events_5min":
                    int(total_events),

                "page_views_5min":
                    int(page_views),

                "product_views_5min":
                    int(product_views),

                "product_clicks_5min":
                    int(product_clicks),

                "add_to_cart_5min":
                    int(add_to_cart),

                "begin_checkout_5min":
                    int(begin_checkout),

                "search_events_5min":
                    int(search_events),

                "is_new_user":
                    1
                    if user_type == "New User"
                    else 0,

                "device_category":
                    device_category,

                "operating_system":
                    operating_system,

                "traffic_source":
                    traffic_source,

                "traffic_medium":
                    traffic_medium
            }


            try:

                with st.spinner(
                    "Scoring customer session..."
                ):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10
                    )


                if response.status_code == 200:

                    result = response.json()

                    probability = result[
                        "purchase_probability_pct"
                    ]

                    intent = result[
                        "intent_segment"
                    ]

                    raw_score = result[
                        "raw_propensity_score"
                    ]


                    # -------------------------------------
                    # RESULT
                    # -------------------------------------

                    st.subheader(
                        "Prediction Result"
                    )

                    metric1, metric2, metric3 = (
                        st.columns(3)
                    )


                    metric1.metric(
                        "Purchase Probability",
                        f"{probability:.2f}%"
                    )


                    metric2.metric(
                        "Intent Segment",
                        intent
                    )


                    metric3.metric(
                        "Raw Propensity Score",
                        f"{raw_score:.4f}"
                    )


                    st.progress(
                        min(
                            probability / 100,
                            1.0
                        )
                    )


                    # -------------------------------------
                    # EXPLANATION
                    # -------------------------------------

                    st.subheader(
                        "How to Interpret This"
                    )

                    st.write(
                        f"""
                        The model estimates approximately
                        **{probability:.2f}% probability**
                        that this session will result in
                        a purchase after the initial
                        5-minute observation window.

                        The session is classified as
                        **{intent}** based on its
                        Random Forest propensity score.
                        """
                    )


                    # -------------------------------------
                    # BUSINESS STRATEGY
                    # -------------------------------------

                    st.subheader(
                        "Recommended Business Strategy"
                    )


                    if intent == "High Intent":

                        st.success(
                            """
                            **High purchase intent detected**

                            • Avoid unnecessary blanket discounts

                            • Focus on checkout completion

                            • Recommend relevant products

                            • Use cart / checkout reminders

                            • Preserve margin where possible
                            """
                        )


                    elif intent == "Medium Intent":

                        st.warning(
                            """
                            **Medium purchase intent detected**

                            • Candidate for controlled campaigns

                            • Test incentives using A/B experiments

                            • Measure incremental uplift

                            • Use campaign economics before scaling

                            • Avoid assuming discount = conversion
                            """
                        )


                    else:

                        st.info(
                            """
                            **Low immediate purchase intent**

                            • Focus on product discovery

                            • Consider remarketing / retargeting

                            • Avoid automatically providing discounts

                            • Test whether promotions actually change behaviour
                            """
                        )


                    # -------------------------------------
                    # RAW REQUEST
                    # -------------------------------------

                    with st.expander(
                        "View API Request / Response"
                    ):

                        st.write(
                            "### Request"
                        )

                        st.json(payload)

                        st.write(
                            "### Response"
                        )

                        st.json(result)


                else:

                    st.error(
                        f"Backend returned HTTP "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )


            except requests.exceptions.RequestException as exc:

                st.error(
                    "Could not connect to FastAPI."
                )

                st.exception(exc)



# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab2:

    st.header(
        "Model Performance"
    )

    st.write(
        "Final performance measured on the "
        "held-out test period."
    )


    st.divider()


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "ROC-AUC",
        "0.952"
    )

    metric2.metric(
        "PR-AUC",
        "0.258"
    )

    metric3.metric(
        "Lift @ Top 5%",
        "15.51×"
    )

    metric4.metric(
        "Recall @ Top 5%",
        "77.54%"
    )


    metric5, metric6, metric7 = (
        st.columns(3)
    )


    metric5.metric(
        "Log Loss",
        "0.0314"
    )

    metric6.metric(
        "Brier Score",
        "0.0076"
    )

    metric7.metric(
        "Test Conversion Rate",
        "0.93%"
    )


    st.divider()


    st.subheader(
        "Targeting Performance"
    )


    lift_data = pd.DataFrame(
        {
            "Targeted Population": [
                "Top 1%",
                "Top 5%",
                "Top 10%",
                "Top 20%",
                "Top 30%"
            ],

            "Conversion Rate (%)": [
                33.514690,
                14.425587,
                8.202785,
                4.340495,
                2.937230
            ],

            "Recall (%)": [
                36.023392,
                77.543860,
                88.187135,
                93.333333,
                94.736842
            ],

            "Lift": [
                36.033583,
                15.509784,
                8.819289,
                4.666717,
                3.157986
            ]
        }
    )


    st.dataframe(
        lift_data,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Lift by Targeting Depth"
    )


    lift_chart = (
        lift_data[
            [
                "Targeted Population",
                "Lift"
            ]
        ]
        .set_index(
            "Targeted Population"
        )
    )


    st.bar_chart(
        lift_chart
    )


    st.info(
        """
        Targeting only the top 5% highest-scoring
        sessions captured approximately **77.5% of
        purchasers** and achieved about **15.5× lift**
        over the overall test conversion rate.
        """
    )


    st.subheader(
        "What the Metrics Mean"
    )


    st.markdown(
        """
        **ROC-AUC:** How well the model ranks purchasers
        above non-purchasers overall.

        **PR-AUC:** Precision-recall performance, especially
        important because purchase events are rare.

        **Lift:** How much higher the conversion rate is
        among targeted sessions compared with the overall
        population.

        **Brier Score:** Measures probability accuracy.
        Lower is better.

        **Log Loss:** Penalizes incorrect probability
        predictions, especially confident mistakes.
        """
    )



# =========================================================
# TAB 3 — INTENT SEGMENTS
# =========================================================

with tab3:

    st.header(
        "Customer Intent Segmentation"
    )

    st.write(
        "Sessions were ranked using the purchase "
        "propensity score."
    )


    segment_df = pd.DataFrame(
        {
            "Intent Segment": [
                "High Intent",
                "Medium Intent",
                "Low Intent"
            ],

            "Sessions": [
                4596,
                13789,
                73541
            ],

            "Purchases": [
                664,
                134,
                57
            ],

            "Session Share (%)": [
                4.999674,
                15.000109,
                80.000218
            ],

            "Purchase Share (%)": [
                77.660819,
                15.672515,
                6.666667
            ],

            "Actual Conversion Rate (%)": [
                14.447346,
                0.971789,
                0.077508
            ],

            "Predicted Conversion Rate (%)": [
                12.928111,
                0.986399,
                0.089116
            ]
        }
    )


    st.dataframe(
        segment_df.round(3),
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Conversion Rate by Intent Segment"
    )


    conversion_chart = (
        segment_df[
            [
                "Intent Segment",
                "Actual Conversion Rate (%)"
            ]
        ]
        .set_index(
            "Intent Segment"
        )
    )


    st.bar_chart(
        conversion_chart
    )


    st.subheader(
        "Purchase Concentration"
    )


    purchase_chart = (
        segment_df[
            [
                "Intent Segment",
                "Purchase Share (%)"
            ]
        ]
        .set_index(
            "Intent Segment"
        )
    )


    st.bar_chart(
        purchase_chart
    )


    st.success(
        """
        **Key finding**

        High Intent represents only about **5% of sessions**
        but contains approximately **77.7% of all purchases**.

        The High Intent group converted at approximately
        **14.45%**, compared with only **0.078%** among
        Low Intent sessions.
        """
    )


    st.subheader(
        "Business Interpretation"
    )


    high_col, medium_col, low_col = (
        st.columns(3)
    )


    with high_col:

        st.markdown(
            """
            ### 🔥 High Intent

            **5% of sessions**

            **77.7% of purchases**

            **14.45% conversion**

            These users already demonstrate
            strong purchase behaviour.

            **Strategy**

            Avoid unnecessary blanket discounts.
            Focus on checkout completion,
            recommendations and customer experience.
            """
        )


    with medium_col:

        st.markdown(
            """
            ### ⚡ Medium Intent

            **15% of sessions**

            **15.7% of purchases**

            **0.97% conversion**

            Some intent exists, but conversion
            remains relatively low.

            **Strategy**

            Use controlled A/B experiments
            to test whether offers create
            incremental conversions.
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

            Focus on product discovery,
            remarketing and engagement instead
            of automatically giving discounts.
            """
        )



# =========================================================
# TAB 4 — CAMPAIGN SIMULATOR
# =========================================================

with tab4:

    st.header(
        "Campaign Profit Simulator"
    )

    st.write(
        """
        Estimate whether a discount campaign could
        become profitable for the Medium Intent segment.

        These values are **business assumptions**,
        not values learned from the GA4 dataset.
        """
    )


    st.divider()


    input1, input2, input3 = (
        st.columns(3)
    )


    with input1:

        gross_margin = st.number_input(
            "Gross Margin per Order (₹)",
            min_value=0.0,
            value=600.0,
            step=50.0
        )


    with input2:

        discount = st.number_input(
            "Discount per Converted Order (₹)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )


    with input3:

        campaign_cost = st.number_input(
            "Campaign Cost per Session (₹)",
            min_value=0.0,
            value=2.0,
            step=0.5
        )


    uplift_pp = st.slider(
        "Expected Incremental Conversion Uplift "
        "(percentage points)",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.05
    )


    # -----------------------------------------------------
    # MEDIUM INTENT BASELINE
    # -----------------------------------------------------

    medium_sessions = 13789

    baseline_conversion_rate = (
        0.971789 / 100
    )


    baseline_orders = (
        medium_sessions
        * baseline_conversion_rate
    )


    new_conversion_rate = (
        baseline_conversion_rate
        + uplift_pp / 100
    )


    new_conversion_rate = min(
        new_conversion_rate,
        1.0
    )


    campaign_orders = (
        medium_sessions
        * new_conversion_rate
    )


    baseline_profit = (
        baseline_orders
        * gross_margin
    )


    campaign_profit = (
        campaign_orders
        * (gross_margin - discount)
        -
        medium_sessions
        * campaign_cost
    )


    incremental_profit = (
        campaign_profit
        -
        baseline_profit
    )


    additional_orders = (
        campaign_orders
        -
        baseline_orders
    )


    result1, result2, result3, result4 = (
        st.columns(4)
    )


    result1.metric(
        "Baseline Orders",
        f"{baseline_orders:,.1f}"
    )


    result2.metric(
        "Expected Campaign Orders",
        f"{campaign_orders:,.1f}",
        delta=f"{additional_orders:,.1f}"
    )


    result3.metric(
        "Baseline Profit",
        f"₹{baseline_profit:,.0f}"
    )


    result4.metric(
        "Incremental Profit",
        f"₹{incremental_profit:,.0f}"
    )


    st.divider()


    st.subheader(
        "Campaign Economics"
    )


    left, right = st.columns(2)


    with left:

        st.metric(
            "Baseline Conversion",
            f"{baseline_conversion_rate * 100:.3f}%"
        )


    with right:

        st.metric(
            "Simulated Campaign Conversion",
            f"{new_conversion_rate * 100:.3f}%",
            delta=f"+{uplift_pp:.2f} pp"
        )


    # -----------------------------------------------------
    # BREAK-EVEN CALCULATION
    # -----------------------------------------------------

    if gross_margin > discount:

        break_even_uplift = (
            (
                discount
                * baseline_conversion_rate
                +
                campaign_cost
            )
            /
            (
                gross_margin
                -
                discount
            )
        )


        break_even_uplift_pp = (
            break_even_uplift * 100
        )


        break_even_conversion = (
            baseline_conversion_rate
            +
            break_even_uplift
        )


        st.subheader(
            "Break-even Analysis"
        )


        b1, b2 = st.columns(2)


        b1.metric(
            "Required Break-even Uplift",
            f"{break_even_uplift_pp:.3f} pp"
        )


        b2.metric(
            "Required Conversion Rate",
            f"{break_even_conversion * 100:.3f}%"
        )


        if uplift_pp > break_even_uplift_pp:

            st.success(
                f"""
                Under these assumptions, the simulated
                campaign is **profitable**.

                Expected incremental profit:
                **₹{incremental_profit:,.0f}**
                """
            )


        elif abs(
            uplift_pp
            -
            break_even_uplift_pp
        ) < 0.01:

            st.warning(
                "The campaign is approximately at break-even."
            )


        else:

            st.error(
                f"""
                Under these assumptions, the campaign
                does **not** break even.

                Expected incremental profit:
                **₹{incremental_profit:,.0f}**
                """
            )


    else:

        st.error(
            """
            Discount must be smaller than the
            gross margin per order for this
            break-even model to be meaningful.
            """
        )


    st.divider()


    st.info(
        """
        Important: this simulator does **not prove**
        that a discount will cause the assumed uplift.

        Real incremental uplift should be measured
        using randomized A/B testing and potentially
        modeled using uplift / causal ML techniques.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Model: Random Forest | "
    "Calibration: Isotonic Regression | "
    "Observation Window: First 5 Minutes"
)