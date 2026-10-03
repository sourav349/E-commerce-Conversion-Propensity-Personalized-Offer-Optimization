# E-commerce Conversion Propensity & Personalized Campaign Optimization

An end-to-end Data Science and Machine Learning project that predicts e-commerce purchase propensity from the first five minutes of customer behavior, segments sessions into purchase-intent groups, explains model decisions using SHAP, serves predictions through FastAPI, provides an interactive Streamlit dashboard, and evaluates campaign profitability.

---

## Project Overview

Most visitors to an e-commerce website do not purchase.

The goal of this project is to identify customers showing strong purchase intent early in their session.

The system observes customer behavior during the:

**First 5 minutes of a session**

and predicts:

**Whether the session will result in a purchase after minute 5**

The complete solution includes:

- Google Analytics 4 e-commerce data
- Google BigQuery
- Exploratory Data Analysis
- Feature Engineering
- Logistic Regression
- Random Forest
- XGBoost
- Probability Calibration
- Intent Segmentation
- Campaign Economics
- SHAP Explainability
- FastAPI Backend
- Streamlit Frontend

---

# 1. Business Problem

E-commerce businesses receive large amounts of traffic, but only a small fraction of sessions generate purchases.

Treating every customer equally can lead to inefficient marketing spend.

For example:

- High-intent customers may purchase without discounts.
- Medium-intent customers may respond to promotions.
- Low-intent customers may not respond even when offered discounts.

The first objective is therefore:

> Who is most likely to purchase?

This is a **purchase propensity problem**.

The second business question is:

> Who will purchase specifically because of a promotion?

This is an **uplift / causal inference problem**.

This project primarily solves the purchase propensity problem and adds a campaign economics simulator to estimate how much incremental uplift would be required for a promotion to become profitable.

---

# 2. Dataset

The project uses Google's public GA4 e-commerce sample dataset in BigQuery:

```text
bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*