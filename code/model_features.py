"""Shared predictor definitions for the Week 10 models.

Both the training script and the Streamlit app import this file, so a
derived predictor is always calculated the same way in training, in the
what-if analysis, and in the contribution calculations.

Raw predictors are the values a user can see or edit. Model predictors are
calculated from them by `add_model_features`.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# KPI 1 — seven-day item demand
# ---------------------------------------------------------------------------
DEMAND_RAW = [
    "ed_arrivals",            # patients who arrived today
    "forecast_arrivals_1d",   # patients expected tomorrow
    "forecast_arrivals_7d",   # patients expected over the next seven days
    "demand_lag1",            # yesterday's demand for the item
    "demand_roll7",           # trailing seven-day average daily demand for the item
    "use_rate_per_arrival",   # units of this item used per patient (item assumption)
    "day_of_week_num",        # 0 = Monday ... 6 = Sunday
    "month",                  # 1-12
    "weekend_flag",           # 1 when today is Saturday or Sunday
    "holiday_flag",           # 1 when today is a holiday
    "surge_flag",             # 1 when today is inside an arrival surge
]
# The demand model uses the raw predictors directly. The item code is left out
# because units-per-patient already identifies each item's scale.
DEMAND_MODEL_NUMERIC = list(DEMAND_RAW)

# ---------------------------------------------------------------------------
# KPI 2 — seven-day shortage probability
# ---------------------------------------------------------------------------
SHORTAGE_RAW = [
    "days_of_supply",         # inventory position ÷ average daily forecast demand
    "lead_time_days",         # supplier lead time for the item
    "closing_on_hand",        # physical stock at the end of the day
    "demand_lag1",            # yesterday's demand for the item
    "demand_roll7",           # trailing seven-day average daily demand
    "forecast_arrivals_7d",   # patients expected over the next seven days
    "avg_arrivals_28d",       # average daily arrivals over the previous 28 days
    "surge_flag",             # 1 when today is inside an arrival surge
    "supplier_event",         # supplier problem on receipts due today
]
SHORTAGE_MODEL_NUMERIC = [
    "days_of_supply", "coverage_gap_days", "stock_cover_ratio",
    "demand_trend", "arrival_ratio", "surge_flag",
]
SHORTAGE_MODEL_CATEGORICAL = ["supplier_event"]
SUPPLIER_EVENTS = ["none", "delayed", "short_shipped", "delayed+short_shipped"]


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every derived model predictor added.

    Ratios use a floor of 1 in the denominator so a zero value cannot
    produce an infinite predictor.
    """
    out = df.copy()
    cols = set(out.columns)
    # Arrival-driven expected demand: the benchmark the demand model is compared with.
    if {"forecast_arrivals_7d", "use_rate_per_arrival"} <= cols:
        out["expected_demand_7d"] = out["forecast_arrivals_7d"] * out["use_rate_per_arrival"]
    if "demand_roll7" in cols:
        out["recent_demand_7d"] = out["demand_roll7"] * 7
    # Shortage predictors (scale-free, so they compare across items).
    if set(SHORTAGE_RAW) - {"supplier_event"} <= cols:
        out["coverage_gap_days"] = out["days_of_supply"] - out["lead_time_days"]
        out["stock_cover_ratio"] = out["closing_on_hand"] / out["recent_demand_7d"].clip(lower=1)
        out["demand_trend"] = out["demand_lag1"] / out["demand_roll7"].clip(lower=1)
        out["arrival_ratio"] = out["forecast_arrivals_7d"] / (out["avg_arrivals_28d"] * 7).clip(lower=1)
    return out
