from datetime import date
from typing import List, Literal

from pydantic import BaseModel, Field


class SessionInput(BaseModel):
    session_date: date

    total_events_5min: int = Field(ge=0)
    page_views_5min: int = Field(ge=0)
    product_views_5min: int = Field(ge=0)
    product_clicks_5min: int = Field(ge=0)
    add_to_cart_5min: int = Field(ge=0)
    begin_checkout_5min: int = Field(ge=0)
    search_events_5min: int = Field(ge=0)

    is_new_user: Literal[0, 1]

    device_category: str
    operating_system: str
    traffic_source: str
    traffic_medium: str


class ShapDriver(BaseModel):
    feature: str
    value: str
    shap_value: float
    direction: Literal["up", "down"]


class PredictionResponse(BaseModel):
    raw_propensity_score: float

    purchase_probability: float
    purchase_probability_pct: float

    intent_segment: str

    explanation_available: bool
    explanation_note: str

    top_positive_drivers: List[ShapDriver]
    top_negative_drivers: List[ShapDriver]