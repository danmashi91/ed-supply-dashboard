"""
ED Supply Decision Support — Version 3 (Predictive Validation Dashboard)

Streamlit multipage app built from docs/app_specification.md
(data version v1.1-2026-10-19).

Run from the app_build/ folder:
    pip install -r requirements.txt
    streamlit run app.py

Design notes
- One file, using st.navigation. The sidebar filters are drawn in this file
  on every run, so they keep their values when the user switches pages.
- Every CSV is loaded once with @st.cache_data. The app only reads data.
- Calculation rules R1–R8 from Section 4 of the spec are implemented in the
  helper functions in the "Calculation rules" section below. Every page uses
  those helpers, so each rule is defined in exactly one place.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent          # app_build/ (next to data/)
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"

PATHS = {
    "arrivals": DATA_DIR / "raw" / "generated_ed_arrivals.csv",
    "items": DATA_DIR / "raw" / "item_assumptions.csv",
    "daily": DATA_DIR / "final" / "inventory_decision_daily.csv",
    "scenario": DATA_DIR / "final" / "scenario_comparison.csv",
    "validation": DATA_DIR / "final" / "validation_results.csv",
}

MODEL_PATHS = {
    "demand": MODEL_DIR / "demand_learned_model.joblib",   # learned comparison model
    "shortage": MODEL_DIR / "shortage_model.joblib",
    "metadata": MODEL_DIR / "model_metadata.json",
    "model_data": DATA_DIR / "modeling" / "model_dataset.csv",
    "importance": DATA_DIR / "modeling" / "global_importance.csv",
    "error_analysis": DATA_DIR / "modeling" / "error_analysis.csv",
}

ASSUMPTION_VERSION = "v1.1-2026-10-19"
FOOTER_TEXT = (
    "Prototype using synthetic data and illustrative assumptions "
    f"(assumption version {ASSUMPTION_VERSION}). Not for real ordering or "
    "clinical decisions. The descriptive risk score is a ranking proxy; the validated "
    "shortage output is a model estimate, not a guaranteed outcome."
)
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SCENARIO_ORDER = ["low", "base", "high"]

st.set_page_config(page_title="ED Supply Decision Support", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached, read-only)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_csv(path: str, has_date: bool) -> pd.DataFrame:
    """Read one CSV. Parse the `date` column as datetime when present."""
    df = pd.read_csv(path)
    if has_date:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load the five CSVs. Stop with a clear message if a file is missing."""
    missing = [str(p.relative_to(APP_ROOT)) for p in PATHS.values() if not p.exists()]
    if missing:
        st.error(
            "These data files were not found next to app.py:\n\n- "
            + "\n- ".join(missing)
            + "\n\nPlace app.py in the app_build/ folder, beside its data/ folder."
        )
        st.stop()
    return {
        "arrivals": load_csv(str(PATHS["arrivals"]), True),
        "items": load_csv(str(PATHS["items"]), False),
        "daily": load_csv(str(PATHS["daily"]), True),
        "scenario": load_csv(str(PATHS["scenario"]), True),
        "validation": load_csv(str(PATHS["validation"]), False),
    }


data = load_all()
ARRIVALS = data["arrivals"]
ITEMS = data["items"]
DAILY = data["daily"]
SCENARIO = data["scenario"]
VALIDATION = data["validation"]


@st.cache_resource(show_spinner="Loading predictive models…")
def load_models() -> tuple[object, object, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the two trained models and their transparent supporting outputs."""
    import json

    missing = [str(p.relative_to(APP_ROOT)) for p in MODEL_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    metadata_preview = json.loads(MODEL_PATHS["metadata"].read_text(encoding="utf-8"))
    # Week 11: KPI 1 uses the arrival formula when evaluation retained it; the
    # learned model file is then kept only as a documented comparison.
    if metadata_preview["demand"].get("retained_type") == "formula":
        demand_model = MF.ArrivalFormulaModel()
    else:
        demand_model = joblib.load(MODEL_PATHS["demand"])
    shortage_model = joblib.load(MODEL_PATHS["shortage"])
    metadata = json.loads(MODEL_PATHS["metadata"].read_text(encoding="utf-8"))
    model_data = pd.read_csv(MODEL_PATHS["model_data"], parse_dates=["date"])
    importance = pd.read_csv(MODEL_PATHS["importance"])
    error_analysis = pd.read_csv(MODEL_PATHS["error_analysis"])
    return demand_model, shortage_model, metadata, model_data, importance, error_analysis

# Lookups for display names (always show item_name, not the code).
ITEM_NAME = dict(zip(ITEMS["item_id"], ITEMS["item_name"]))
ITEM_CATEGORY = dict(zip(ITEMS["item_id"], ITEMS["category"]))
ITEM_INFO = ITEMS.set_index("item_id")

# 7-day trailing rolling average on the full arrival series, so the first
# days of a narrowed date range still have a full 7-day window.
ARRIVALS = ARRIVALS.sort_values("date").reset_index(drop=True)
ARRIVALS["rolling_7d"] = ARRIVALS["ed_arrivals"].rolling(7, min_periods=1).mean()

DATA_MIN = ARRIVALS["date"].min().date()
DATA_MAX = ARRIVALS["date"].max().date()


# ---------------------------------------------------------------------------
# Calculation rules (Section 4) — the only place these formulas live
# ---------------------------------------------------------------------------
def arrivals_one_row_per_date(arr: pd.DataFrame) -> pd.DataFrame:
    """R1: arrival metrics always come from generated_ed_arrivals.csv
    (one row per date), never from the item-level daily table."""
    return arr.drop_duplicates("date")


def arrival_forecast_by_date(start: date, end: date) -> pd.DataFrame:
    """R1/R3: de-duplicate the daily table to one row per date to get the
    1-day arrival forecast. Not filtered by item, so item filters cannot
    change it."""
    mask = (DAILY["date"].dt.date >= start) & (DAILY["date"].dt.date <= end)
    return DAILY.loc[mask, ["date", "ed_arrivals", "forecast_arrivals_1d"]].drop_duplicates("date")


def wape(start: date, end: date) -> float | None:
    """R3: sum(|ed_arrivals − forecast_arrivals_1d|) ÷ sum(ed_arrivals),
    one row per date."""
    d = arrival_forecast_by_date(start, end)
    denom = d["ed_arrivals"].sum()
    if denom == 0:
        return None
    return (d["ed_arrivals"] - d["forecast_arrivals_1d"]).abs().sum() / denom


def fill_rate(df: pd.DataFrame) -> float | None:
    """R2: sum(min(issued, requested)) ÷ sum(requested), row by row."""
    requested = df["item_demand_requested"].sum()
    if requested == 0:
        return None
    filled = df[["item_issued_qty", "item_demand_requested"]].min(axis=1).sum()
    return filled / requested


def backorder_days(df: pd.DataFrame) -> int:
    """R2b: count of rows where backorder_qty > 0."""
    return int((df["backorder_qty"] > 0).sum())


def high_risk_count(df: pd.DataFrame, threshold: float) -> int:
    """R4: count of rows where stockout_risk_proxy ≥ threshold."""
    return int((df["stockout_risk_proxy"] >= threshold).sum())


def coverage_below_lead_time(df: pd.DataFrame) -> pd.Series:
    """R5: days_of_supply < lead_time_days (boolean flag)."""
    return df["days_of_supply"] < df["lead_time_days"]


def sort_risk_queue(df: pd.DataFrame) -> pd.DataFrame:
    """R6: risk descending, then days of supply ascending."""
    return df.sort_values(["stockout_risk_proxy", "days_of_supply"], ascending=[False, True])


def order_spend(df: pd.DataFrame) -> float:
    """R7: sum(order_placed_qty × unit_cost). Never sums estimated_order_cost."""
    return float((df["order_placed_qty"] * df["unit_cost"]).sum())


def supplier_disruptions(df: pd.DataFrame) -> int:
    """R8: count of rows where supplier_event ≠ "none"."""
    return int((df["supplier_event"] != "none").sum())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def fmt_usd0(x: float) -> str:
    return f"${x:,.0f}"


def fmt_usd2(x: float) -> str:
    return f"${x:,.2f}"


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


def footer() -> None:
    """Footer banner shown on every page."""
    st.divider()
    st.info(FOOTER_TEXT, icon="ℹ️")


def stop_with(message: str) -> None:
    """Friendly message for empty filter results, then stop this page."""
    st.info(message, icon="🔎")
    footer()
    st.stop()


def readable(col: str) -> str:
    """snake_case column name -> readable header."""
    return col.replace("_", " ").capitalize()


PREDICTOR_LABELS = {
    "ed_arrivals": "Current ED arrivals",
    "forecast_arrivals_1d": "One-day arrival forecast",
    "forecast_arrivals_7d": "Seven-day arrival forecast",
    "avg_arrivals_28d": "Average daily arrivals, past 28 days",
    "demand_lag1": "Previous-day item demand",
    "demand_roll7": "Trailing seven-day average demand",
    "use_rate_per_arrival": "Units used per arrival",
    "day_of_week_num": "Day of week",
    "month": "Month",
    "weekend_flag": "Weekend indicator",
    "holiday_flag": "Holiday indicator",
    "surge_flag": "Demand-surge indicator",
    "closing_on_hand": "Closing stock",
    "days_of_supply": "Days of supply",
    "lead_time_days": "Supplier lead time",
    "supplier_event": "Supplier event",
}

# Why each predictor is relevant (rubric item 2a). Shown on the Predictive Models page.
PREDICTOR_REASONS = {
    "demand": {
        "Seven-day arrival forecast": "More patients over the coming week means more supplies used. The retained method scales demand directly with it.",
        "Units used per arrival": "Sets each item's scale: a patient uses far more gloves than specimen cups.",
        "Recent demand, surge and calendar fields (learned models only)": "Tested in the learned comparison models. They did not improve accuracy over the formula on 2024 validation data.",
    },
    "shortage": {
        "Days of supply and supplier lead time": "Together show whether stock lasts until a new order could arrive (coverage gap = days of supply − lead time).",
        "Closing stock": "Physical stock today; compared with recent weekly demand, it shows how many days of real cover remain.",
        "Previous-day and trailing seven-day demand": "A rising trend can use up stock faster than planned.",
        "Seven-day arrival forecast vs. 28-day average arrivals": "A forecast above the recent norm signals extra pressure on stock.",
        "Demand-surge indicator": "Surges consume stock faster than the ordering plan assumed.",
        "Supplier event": "Late or short deliveries leave less stock than planned.",
    },
}


def load_model_features():
    """Import code/model_features.py, the predictor definitions shared with training."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_features", APP_ROOT / "code" / "model_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MF = load_model_features()


def fmt_value(value: object) -> str:
    """Readable value for summaries: whole numbers without decimals, others with two."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


def model_prediction(model: object, row: pd.DataFrame, details: dict, probability: bool) -> float:
    """Derive model predictors from the raw row, then return one nonnegative
    regression prediction or class-1 probability."""
    X = MF.add_model_features(row)[details["model_predictors"]]
    if probability:
        return float(model.predict_proba(X)[:, 1][0])
    return max(0.0, float(model.predict(X)[0]))


def reference_replacement_contributions(
    model: object,
    row: pd.DataFrame,
    details: dict,
    probability: bool,
) -> pd.DataFrame:
    """Approximate local contributions through one-predictor reference replacement.

    For each raw predictor, compare the full prediction with a prediction where
    that predictor alone is replaced by its typical training value. The values
    explain model sensitivity for this record and are not causal effects.
    Interactions mean the contributions need not add exactly to the prediction.
    """
    raw = details["raw_predictors"]
    reference = details["reference_values"]
    full = model_prediction(model, row[raw], details, probability)
    records = []
    for feature in raw:
        altered = row[raw].copy()
        altered[feature] = reference[feature]
        without = model_prediction(model, altered, details, probability)
        records.append({
            "predictor": feature,
            "Predictor": PREDICTOR_LABELS.get(feature, readable(feature)),
            "value": fmt_value(row[feature].iloc[0]),
            "typical": fmt_value(reference[feature]),
            "contribution": full - without,
        })
    return pd.DataFrame(records).sort_values("contribution", key=lambda s: s.abs(), ascending=False)


def contribution_chart(contrib: pd.DataFrame, probability: bool) -> go.Figure:
    """Horizontal bar chart of the ten largest contributions to one prediction."""
    chart = contrib.head(10).sort_values("contribution").copy()
    chart["Display contribution"] = chart["contribution"] * (100 if probability else 1)
    chart["Direction"] = np.where(chart["contribution"] >= 0, "Pushes prediction up", "Pushes prediction down")
    chart["label"] = chart["Predictor"] + " = " + chart["value"]
    fig = px.bar(
        chart, x="Display contribution", y="label", orientation="h", color="Direction",
        color_discrete_map={"Pushes prediction up": "#E45756", "Pushes prediction down": "#54A24B"},
        hover_data={"typical": True, "label": False},
        labels={
            "Display contribution": "Contribution (percentage points)" if probability else "Contribution (units)",
            "label": "", "typical": "Typical training value",
        },
    )
    fig.update_layout(height=400, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1, title=None))
    return fig


def contribution_sentence(contrib: pd.DataFrame, probability: bool) -> str:
    """Plain-language description of the strongest contributions (direction and size)."""
    unit = "percentage points" if probability else "units"
    scale = 100 if probability else 1

    def describe(rows: pd.DataFrame) -> str:
        return ", ".join(
            f"{r.Predictor} at {r.value} vs. a typical {r.typical} ({r.contribution * scale:+,.1f} {unit})"
            for r in rows.itertuples()
        )

    positive = contrib[contrib["contribution"] > 0].head(2)
    negative = contrib[contrib["contribution"] < 0].head(2)
    parts = []
    if not positive.empty:
        parts.append(f"Pushing the prediction up: {describe(positive)}.")
    if not negative.empty:
        parts.append(f"Pushing it down: {describe(negative)}.")
    return " ".join(parts) or "No single predictor has a material contribution for this prediction."


# ---------------------------------------------------------------------------
# Sidebar filters (all pages)
# ---------------------------------------------------------------------------
ss = st.session_state

st.sidebar.header("Filters")

date_value = st.sidebar.date_input(
    "Date range",
    value=(DATA_MIN, DATA_MAX),
    min_value=DATA_MIN,
    max_value=DATA_MAX,
    key="date_range",
    format="YYYY-MM-DD",
)

# Category multiselect (default all).
all_categories = list(dict.fromkeys(ITEMS["category"]))  # keep file order, unique
if "categories" not in ss:
    ss["categories"] = all_categories
selected_categories = st.sidebar.multiselect("Category", all_categories, key="categories")

# Item multiselect, limited to the chosen categories (default all).
# When the category choice changes, the item list resets to all items in the
# chosen categories; otherwise any item no longer allowed is dropped.
item_options = [i for i in ITEMS["item_id"] if ITEM_CATEGORY[i] in selected_categories]
if "items" not in ss or ss.get("_prev_categories") != selected_categories:
    ss["items"] = list(item_options)
else:
    ss["items"] = [i for i in ss["items"] if i in item_options]
ss["_prev_categories"] = list(selected_categories)
selected_items = st.sidebar.multiselect(
    "Item", item_options, key="items", format_func=lambda i: ITEM_NAME[i]
)

threshold = st.sidebar.slider(
    "Risk threshold", min_value=0.00, max_value=1.00, value=0.50, step=0.05, key="threshold",
    help="Item-days with a risk score at or above this value count as high risk.",
)

# A date range picker returns one date while the user is still choosing.
if not isinstance(date_value, (tuple, list)) or len(date_value) != 2:
    st.sidebar.warning("Pick an end date to complete the date range.")
    stop_with("Choose both a start and an end date in the sidebar to see the dashboard.")
START, END = date_value

# Filtered views.
ARR_F = arrivals_one_row_per_date(
    ARRIVALS[(ARRIVALS["date"].dt.date >= START) & (ARRIVALS["date"].dt.date <= END)]
)  # date filter only: item filters never change arrival numbers (R1)
DAILY_F = DAILY[
    (DAILY["date"].dt.date >= START)
    & (DAILY["date"].dt.date <= END)
    & (DAILY["item_id"].isin(selected_items))
].copy()
DAILY_F["item_name"] = DAILY_F["item_id"].map(ITEM_NAME)
DAILY_F["category"] = DAILY_F["item_id"].map(ITEM_CATEGORY)

st.sidebar.caption(
    f"{len(ARR_F):,} days · {len(selected_items)} of {len(ITEMS)} items · "
    f"{len(DAILY_F):,} item-day rows"
)


def as_of_picker(key: str, label: str = "As-of date") -> date:
    """Date picker bounded by the sidebar range. Defaults to the last date in
    the range, and resets to it whenever the sidebar range changes."""
    range_key = f"_{key}_range"
    if key not in ss or ss.get(range_key) != (START, END):
        ss[key] = END
    ss[range_key] = (START, END)
    return st.date_input(label, min_value=START, max_value=END, key=key, format="YYYY-MM-DD")


def item_picker(key: str) -> str:
    """Single-item selector, limited to the sidebar item list."""
    if ss.get(key) not in selected_items:
        ss[key] = selected_items[0]
    return st.selectbox("Item", selected_items, key=key, format_func=lambda i: ITEM_NAME[i])


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Overview")
    st.caption("What needs attention?")

    # --- KPI row (two rows of 4) ---
    total_arr = ARR_F["ed_arrivals"].sum()
    avg_arr = ARR_F["ed_arrivals"].mean()
    surge_days = int(ARR_F["surge_flag"].sum())
    has_items = not DAILY_F.empty

    r1 = st.columns(4)
    r1[0].metric("Total ED arrivals", fmt_int(total_arr))
    r1[1].metric("Average daily arrivals", f"{avg_arr:,.1f}")
    r1[2].metric("Arrival forecast WAPE", fmt_pct(wape(START, END)),
                 help="Sum of |actual − 1-day forecast| ÷ sum of actual arrivals, one row per date.")
    r1[3].metric("Surge days", f"{surge_days:,}")

    r2 = st.columns(4)
    r2[0].metric("Fill rate", fmt_pct(fill_rate(DAILY_F)) if has_items else "—",
                 help="Sum of min(issued, requested) ÷ sum of requested, row by row.")
    r2[1].metric("Backorder days", f"{backorder_days(DAILY_F):,}" if has_items else "—",
                 help="Item-days with a backorder above zero.")
    r2[2].metric("High-risk item-days", f"{high_risk_count(DAILY_F, threshold):,}" if has_items else "—",
                 help=f"Item-days with risk score ≥ {threshold:.2f}.")
    r2[3].metric("Order spend", fmt_usd0(order_spend(DAILY_F)) if has_items else "—",
                 help="Sum of units actually ordered × unit cost.")
    if not has_items:
        st.caption("Item KPIs show — because no items are selected. Arrival KPIs do not depend on item filters.")

    # --- Chart A: daily arrivals + rolling average, holidays and surges ---
    st.subheader("Daily ED arrivals")
    fig = go.Figure()
    fig.add_scatter(x=ARR_F["date"], y=ARR_F["ed_arrivals"], mode="lines",
                    name="Daily ED arrivals", line=dict(width=1, color="#4C78A8"))
    fig.add_scatter(x=ARR_F["date"], y=ARR_F["rolling_7d"], mode="lines",
                    name="7-day rolling average", line=dict(width=2.5, color="#1B2A49"))
    hol = ARR_F[ARR_F["holiday_flag"] == 1]
    fig.add_scatter(x=hol["date"], y=hol["ed_arrivals"], mode="markers", name="Holiday",
                    marker=dict(symbol="star", size=12, color="#E45756"),
                    hovertemplate="Holiday %{x|%Y-%m-%d}<br>Arrivals %{y}<extra></extra>")
    # Shade surge days: one rectangle per run of consecutive surge dates.
    surge = ARR_F.loc[ARR_F["surge_flag"] == 1, "date"].sort_values()
    if not surge.empty:
        run_id = (surge.diff() != pd.Timedelta(days=1)).cumsum()
        for _, run in surge.groupby(run_id):
            fig.add_vrect(x0=run.min() - pd.Timedelta(hours=12), x1=run.max() + pd.Timedelta(hours=12),
                          fillcolor="#F58518", opacity=0.22, line_width=0, layer="below")
        # Legend entry for the shading.
        fig.add_scatter(x=[None], y=[None], mode="markers", name="Surge day (shaded)",
                        marker=dict(symbol="square", size=12, color="#F58518", opacity=0.4))
    fig.update_layout(height=380, margin=dict(t=10, b=10), yaxis_title="Arrivals",
                      xaxis_title=None, legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # --- Chart B: average arrivals by day of week ---
    st.subheader("Average arrivals by day of week")
    dow = (ARR_F.groupby("day_of_week")["ed_arrivals"].mean()
           .reindex(WEEKDAY_ORDER).dropna().reset_index())
    fig_b = px.bar(dow, x="day_of_week", y="ed_arrivals", text_auto=".1f",
                   labels={"day_of_week": "Day of week", "ed_arrivals": "Average arrivals"},
                   category_orders={"day_of_week": WEEKDAY_ORDER})
    fig_b.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_b, width="stretch")

    # --- Risk queue for one as-of date ---
    st.subheader("Risk queue")
    if DAILY_F.empty:
        st.info("No items match the current filters, so the risk queue is empty. "
                "Select at least one category and item in the sidebar.", icon="🔎")
        return
    as_of = as_of_picker("asof_overview")
    q = DAILY_F[DAILY_F["date"].dt.date == as_of].copy()
    if q.empty:
        st.info(f"No item rows for {as_of}.", icon="🔎")
        return
    q["below_lt"] = coverage_below_lead_time(q)
    q = sort_risk_queue(q)
    table = pd.DataFrame({
        "Item": q["item_name"],
        "Category": q["category"],
        "Inventory position": q["inventory_position"],
        "Days of supply": q["days_of_supply"],
        "Lead time (days)": q["lead_time_days"],
        "Coverage below lead time": q["below_lt"].map({True: "⚠️ Yes", False: "No"}),
        "Risk score": q["stockout_risk_proxy"],
        "Recommended action": q["recommended_action"],
        "Recommended order qty": q["recommended_order_qty"],
        "Estimated order cost": q["estimated_order_cost"],
    }).reset_index(drop=True)

    def highlight(row: pd.Series) -> list[str]:
        on = row["Risk score"] >= threshold
        return ["background-color: rgba(228, 87, 86, 0.22)" if on else ""] * len(row)

    styled = (table.style.apply(highlight, axis=1)
              .format({"Inventory position": "{:,.0f}", "Days of supply": "{:.2f}",
                       "Risk score": "{:.4f}", "Recommended order qty": "{:,.0f}",
                       "Estimated order cost": "${:,.2f}"}))
    st.dataframe(styled, hide_index=True, width="stretch")
    n_hi = int((table["Risk score"] >= threshold).sum())
    st.caption(f"{as_of}: {n_hi} of {len(table)} items at or above the risk threshold "
               f"{threshold:.2f} (highlighted). Sorted by risk score (high to low), then days of supply (low to high).")


# ---------------------------------------------------------------------------
# Page 2 — Item Explorer
# ---------------------------------------------------------------------------
def page_item_explorer() -> None:
    st.title("Item Explorer")
    st.caption("Why is this item flagged?")
    if DAILY_F.empty or not selected_items:
        stop_with("No items match the current filters. Select at least one category and item in the sidebar.")

    c1, c2 = st.columns(2)
    with c1:
        item = item_picker("item_explorer_item")
    with c2:
        as_of = as_of_picker("asof_item")

    df = DAILY_F[DAILY_F["item_id"] == item].sort_values("date")
    if df.empty:
        stop_with("No rows for this item in the selected date range.")
    info = ITEM_INFO.loc[item]
    name = ITEM_NAME[item]
    lead_time = int(info["lead_time_days"])
    avg_demand = df["item_demand_requested"].mean()
    safety_stock = info["safety_stock_days"] * avg_demand
    today = df[df["date"].dt.date == as_of]
    row = today.iloc[0] if not today.empty else None

    # --- KPI row ---
    k = st.columns(6)
    k[0].metric("Current days of supply", f"{row['days_of_supply']:.2f}" if row is not None else "—",
                help=f"On the as-of date ({as_of}).")
    k[1].metric("Lead time", f"{lead_time} days")
    k[2].metric("Avg daily demand", f"{avg_demand:,.1f}", help="Average units requested per day in the range.")
    k[3].metric("High-risk days", f"{high_risk_count(df, threshold):,}", help=f"Risk score ≥ {threshold:.2f}.")
    k[4].metric("Supplier disruptions", f"{supplier_disruptions(df):,}")
    k[5].metric("Order spend", fmt_usd0(order_spend(df)))

    # --- Explanation box ---
    if row is not None:
        qty = int(row["recommended_order_qty"])
        cases = qty / int(row["case_pack"])
        cases_txt = f"{cases:,.0f}" if float(cases).is_integer() else f"{cases:,.2f}"
        text = (f"On {as_of:%Y-%m-%d}, {name} had {row['days_of_supply']:.2f} days of supply "
                f"against a {lead_time}-day lead time. Risk score {row['stockout_risk_proxy']:.4f}. "
                f"Recommended action: {row['recommended_action']}, order {qty:,} units "
                f"({cases_txt} cases) costing {fmt_usd2(row['estimated_order_cost'])}.")
        # Supplier issue on the as-of date or in the 7 days before it (full history,
        # so an event just before the range start is still reported).
        window = DAILY[(DAILY["item_id"] == item)
                       & (DAILY["date"].dt.date >= as_of - timedelta(days=7))
                       & (DAILY["date"].dt.date <= as_of)
                       & (DAILY["supplier_event"] != "none")].sort_values("date")
        if not window.empty:
            last = window.iloc[-1]
            text += f" Recent supplier issue: {last['supplier_event']} on {last['date']:%Y-%m-%d}."
        surge_today = ARRIVALS.loc[ARRIVALS["date"].dt.date == as_of, "surge_flag"]
        if not surge_today.empty and int(surge_today.iloc[0]) == 1:
            text += " ED arrivals are in a surge period."
        st.info(text, icon="📝")

    # --- Chart C: inventory over time ---
    st.subheader("Inventory over time")
    fig_c = go.Figure()
    fig_c.add_scatter(x=df["date"], y=df["closing_on_hand"], mode="lines", name="Closing on hand")
    fig_c.add_scatter(x=df["date"], y=df["inventory_position"], mode="lines", name="Inventory position")
    fig_c.add_hline(y=safety_stock, line_dash="dash", line_color="#E45756",
                    annotation_text=f"Safety stock ≈ {safety_stock:,.0f} "
                                    f"({int(info['safety_stock_days'])} days × {avg_demand:,.1f}/day)",
                    annotation_position="top left", annotation_bgcolor="rgba(255,255,255,0.85)")
    fig_c.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title="Units",
                        legend=dict(orientation="h", y=1.1), hovermode="x unified")
    st.plotly_chart(fig_c, width="stretch")

    # --- Chart D: demand requested vs issued, receipts, supplier events ---
    st.subheader("Demand, issues and receipts")
    fig_d = go.Figure()
    fig_d.add_bar(x=df["date"], y=df["receipts_qty"], name="Receipts", marker_color="#B8C7DA", opacity=0.8)
    fig_d.add_scatter(x=df["date"], y=df["item_demand_requested"], mode="lines", name="Requested",
                      line=dict(color="#4C78A8"))
    fig_d.add_scatter(x=df["date"], y=df["item_issued_qty"], mode="lines", name="Issued",
                      line=dict(color="#54A24B", dash="dot"))
    ev = df[df["supplier_event"] != "none"]
    if not ev.empty:
        fig_d.add_scatter(
            x=ev["date"], y=ev[["receipts_qty", "item_demand_requested"]].max(axis=1), mode="markers",
            name="Supplier event", marker=dict(symbol="x", size=11, color="#E45756", line=dict(width=1)),
            customdata=ev[["supplier_event", "receipts_qty"]].to_numpy(),
            hovertemplate="%{x|%Y-%m-%d}<br>Supplier event: %{customdata[0]}"
                          "<br>Receipts: %{customdata[1]}<extra></extra>")
    fig_d.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title="Units", barmode="overlay",
                        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_d, width="stretch")

    # --- Chart D2: risk over time ---
    st.subheader("Stockout risk score over time")
    fig_d2 = px.line(df, x="date", y="stockout_risk_proxy",
                     labels={"date": "", "stockout_risk_proxy": "Risk score (ranking proxy)"})
    fig_d2.add_hline(y=threshold, line_dash="dash", line_color="#E45756",
                     annotation_text=f"Threshold {threshold:.2f}", annotation_position="top left")
    fig_d2.update_layout(height=300, margin=dict(t=10, b=10), yaxis_range=[0, 1.05])
    st.plotly_chart(fig_d2, width="stretch")

    # --- Chart E: days of supply over time ---
    st.subheader("Days of supply over time")
    fig_e = px.line(df, x="date", y="days_of_supply", labels={"date": "", "days_of_supply": "Days of supply"})
    fig_e.add_hline(y=lead_time, line_dash="dash", line_color="#E45756",
                    annotation_text=f"Lead time {lead_time} days", annotation_position="top left")
    fig_e.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig_e, width="stretch")


# ---------------------------------------------------------------------------
# Page 3 — Trends & Mix
# ---------------------------------------------------------------------------
def page_trends() -> None:
    st.title("Trends & Mix")
    st.caption("What drives demand and cost?")

    if DAILY_F.empty:
        st.info("No items match the current filters, so item charts are hidden. "
                "Arrival chart H is still shown below.", icon="🔎")
    else:
        df = DAILY_F.copy()
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df["spend"] = df["order_placed_qty"] * df["unit_cost"]  # R7 per row

        # --- Chart F: monthly units requested by category ---
        st.subheader("Monthly units requested by category")
        f = df.groupby(["month", "category"], as_index=False)["item_demand_requested"].sum()
        fig_f = px.bar(f, x="month", y="item_demand_requested", color="category", barmode="stack",
                       labels={"month": "Month", "item_demand_requested": "Units requested", "category": "Category"})
        fig_f.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig_f, width="stretch")

        left, right = st.columns(2)
        # --- Chart G: order spend by item (R7) ---
        with left:
            st.subheader("Order spend by item")
            g = df.groupby("item_name", as_index=False)["spend"].sum().sort_values("spend")
            fig_g = px.bar(g, x="spend", y="item_name", orientation="h",
                           text=g["spend"].map(fmt_usd0),
                           labels={"spend": "Order spend (USD)", "item_name": ""})
            fig_g.update_layout(height=380, margin=dict(t=10, b=10), xaxis_tickprefix="$")
            st.plotly_chart(fig_g, width="stretch")
        # --- Chart G2: high-risk item-days by month (all months shown) ---
        with right:
            st.subheader("High-risk item-days by month")
            all_months = pd.period_range(START, END, freq="M").to_timestamp()
            g2 = (df[df["stockout_risk_proxy"] >= threshold].groupby("month").size()
                  .reindex(all_months, fill_value=0).rename_axis("month").reset_index(name="count"))
            fig_g2 = px.bar(g2, x="month", y="count",
                            labels={"month": "Month", "count": "High-risk item-days"})
            fig_g2.update_traces(hovertemplate="%{x|%b %Y}: %{y} item-days with risk ≥ " + f"{threshold:.2f}<extra></extra>")
            fig_g2.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig_g2, width="stretch")

    # --- Chart H: average daily arrivals by day type (R1: arrivals table) ---
    st.subheader("Average daily arrivals: weekday vs weekend vs holiday")
    arr = ARR_F.copy()
    # Holiday takes precedence; weekend next; otherwise weekday.
    arr["day_type"] = "Weekday"
    arr.loc[arr["weekend_flag"] == 1, "day_type"] = "Weekend"
    arr.loc[arr["holiday_flag"] == 1, "day_type"] = "Holiday"
    h = (arr.groupby("day_type").agg(avg=("ed_arrivals", "mean"), days=("ed_arrivals", "size"))
         .reindex(["Weekday", "Weekend", "Holiday"]).dropna().reset_index())
    fig_h = px.bar(h, x="day_type", y="avg", text_auto=".1f", hover_data={"days": True},
                   labels={"day_type": "", "avg": "Average daily arrivals", "days": "Days"})
    fig_h.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_h, width="stretch")

    # --- Per-item summary table ---
    if DAILY_F.empty:
        return
    st.subheader("Per-item summary for the filtered range")
    rows = []
    for item_id, g in DAILY_F.groupby("item_id", sort=False):
        rows.append({
            "Item": ITEM_NAME[item_id],
            "Category": ITEM_CATEGORY[item_id],
            "Total requested": int(g["item_demand_requested"].sum()),
            "Fill rate": fill_rate(g),
            "Backorder days": backorder_days(g),
            "Avg days of supply": g["days_of_supply"].mean(),
            "High-risk days": high_risk_count(g, threshold),
            "Supplier disruptions": supplier_disruptions(g),
            "Order spend": order_spend(g),
        })
    summary = pd.DataFrame(rows)
    st.dataframe(
        summary.style.format({
            "Total requested": "{:,.0f}",
            "Fill rate": lambda v: fmt_pct(None if pd.isna(v) else v),
            "Avg days of supply": "{:.2f}",
            "Order spend": "${:,.0f}",
        }),
        hide_index=True, width="stretch",
    )


# ---------------------------------------------------------------------------
# Page 4 — Scenario Comparison
# ---------------------------------------------------------------------------
def page_scenarios() -> None:
    st.title("Scenario Comparison")
    st.caption("What if demand is higher or lower?")
    if not selected_items:
        stop_with("No items match the current filters. Select at least one category and item in the sidebar.")

    c1, c2 = st.columns(2)
    with c1:
        as_of = as_of_picker("asof_scenario")
    with c2:
        item = item_picker("scenario_item")

    s = SCENARIO[(SCENARIO["date"].dt.date == as_of) & (SCENARIO["item_id"] == item)].copy()
    if s.empty:
        stop_with(f"No scenario rows for {ITEM_NAME[item]} on {as_of}.")
    s["scenario"] = pd.Categorical(s["scenario"], categories=SCENARIO_ORDER, ordered=True)
    s = s.sort_values("scenario")
    unit_cost = float(ITEM_INFO.loc[item, "unit_cost"])          # from item assumptions
    s["estimated_cost"] = s["recommended_order_qty"] * unit_cost

    table = pd.DataFrame({
        "Scenario": s["scenario"].astype(str),
        "Demand multiplier": s["demand_multiplier"],
        "7-day item demand": s["forecast_item_demand_7d"],
        "Inventory position": s["inventory_position"],
        "Days of supply": s["days_of_supply"],
        "Risk score": s["stockout_risk_proxy"],
        "Recommended order qty": s["recommended_order_qty"],
        "Estimated cost": s["estimated_cost"],
    })
    st.subheader(f"{ITEM_NAME[item]} on {as_of:%Y-%m-%d}")
    st.dataframe(
        table.style.format({"Demand multiplier": "{:.2f}", "7-day item demand": "{:,.1f}",
                            "Inventory position": "{:,.0f}", "Days of supply": "{:.2f}",
                            "Risk score": "{:.4f}", "Recommended order qty": "{:,.0f}",
                            "Estimated cost": "${:,.2f}"}),
        hide_index=True, width="stretch",
    )

    # --- Chart I: grouped bars; days of supply on a second axis so both are readable ---
    st.subheader("Recommended order and days of supply by scenario")
    fig = go.Figure()
    fig.add_bar(x=table["Scenario"], y=table["Recommended order qty"], name="Recommended order qty (units, left axis)",
                offsetgroup=0, marker_color="#4C78A8", text=table["Recommended order qty"].map("{:,.0f}".format),
                textposition="outside")
    fig.add_bar(x=table["Scenario"], y=table["Days of supply"], name="Days of supply (right axis)",
                offsetgroup=1, yaxis="y2", marker_color="#F58518", text=table["Days of supply"].map("{:.2f}".format),
                textposition="outside")
    fig.update_layout(
        barmode="group", height=380, margin=dict(t=30, b=10),
        xaxis=dict(title="Scenario", categoryorder="array", categoryarray=SCENARIO_ORDER),
        yaxis=dict(title="Recommended order qty (units)", rangemode="tozero"),
        yaxis2=dict(title="Days of supply", overlaying="y", side="right", rangemode="tozero", showgrid=False),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Physical stock is the same in all scenarios. Only forecast demand changes.")


# ---------------------------------------------------------------------------
# Page 5 — Predictive Models & What-If Analysis (Week 10)
# ---------------------------------------------------------------------------
def page_predictive_models() -> None:
    st.title("Predictive Model Validation & What-If Analysis")
    st.caption("How credible is the model, where does it struggle, and what changes under a plausible scenario?")

    try:
        demand_model, shortage_model, metadata, model_data, importance, error_analysis = load_models()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python code/train_predictive_models.py")
        footer()
        st.stop()

    st.info(
        "Both models are trained on synthetic data. **Overall importance** describes what the model relies "
        "on across the whole 2025 test period. **Individual contributions** explain one prediction for one "
        "item on one date. Neither is evidence that a predictor causes the outcome.",
        icon="ℹ️",
    )

    # ------------------------------------------------------------------
    # Recommended actions (added after user testing: the participant asked for
    # simple recommendations on which items need attention and when to order)
    # ------------------------------------------------------------------
    st.subheader("Recommended actions")
    rec_pool = model_data[model_data["item_id"].isin(selected_items)]
    rec_pool = rec_pool[(rec_pool["date"].dt.date >= START) & (rec_pool["date"].dt.date <= END)]
    if rec_pool.empty:
        st.info("No model rows match the sidebar filters. Model rows end on 2025-12-24.", icon="🔎")
    else:
        rec_dates = sorted(rec_pool["date"].dt.date.unique())
        rec_date = st.selectbox("Review date", rec_dates, index=len(rec_dates) - 1, key="rec_date",
                                format_func=lambda d: d.strftime("%Y-%m-%d"))
        rows = rec_pool[rec_pool["date"].dt.date == rec_date]
        stock = DAILY[DAILY["date"].dt.date == rec_date].set_index("item_id")
        s_thr = float(metadata["shortage"].get("decision_threshold", 0.5))
        recs = []
        for _, r in rows.iterrows():
            one = r.to_frame().T.reset_index(drop=True)
            p_short = model_prediction(shortage_model, one[metadata["shortage"]["raw_predictors"]], metadata["shortage"], True)
            demand7 = model_prediction(demand_model, one[metadata["demand"]["raw_predictors"]], metadata["demand"], False)
            pos = float(stock.loc[r["item_id"], "inventory_position"])
            pack = int(stock.loc[r["item_id"], "case_pack"])
            gap = max(0.0, demand7 - pos)
            order_units = int(np.ceil(gap / pack) * pack) if gap > 0 else 0
            if p_short >= s_thr:
                action = "🔴 Review now: check stock and supplier; order or expedite if confirmed"
                rank = 0
            elif order_units > 0:
                action = "🟠 Order this week: stock will not cover next week's demand"
                rank = 1
            else:
                action = "🟢 No action needed: stock covers next week"
                rank = 2
            recs.append({"rank": rank, "p": p_short, "Item": ITEM_NAME.get(r["item_id"], r["item_id"]),
                         "Recommended action": action,
                         "Shortage probability": f"{p_short:.1%}",
                         "Expected use, next 7 days": f"{demand7:,.0f}",
                         "Stock incl. open orders": f"{pos:,.0f}",
                         "Suggested order (units)": f"{order_units:,}" + (f" ({order_units // pack} case{'s' if order_units // pack != 1 else ''})" if order_units else "")})
        rec_df = pd.DataFrame(recs).sort_values(["rank", "p"], ascending=[True, False])
        n_red, n_orange = int((rec_df["rank"] == 0).sum()), int((rec_df["rank"] == 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Review now", n_red)
        c2.metric("Order this week", n_orange)
        c3.metric("No action needed", len(rec_df) - n_red - n_orange)
        st.dataframe(rec_df.drop(columns=["rank", "p"]), hide_index=True, width="stretch")
        st.caption(
            f"How these are made: **Review now** = shortage probability at or above the {s_thr:.1%} screening "
            "threshold. **Order this week** = expected use over the next 7 days is more than stock on hand plus open "
            "orders; the suggested order covers the gap, rounded up to whole case packs. These are model-based "
            "suggestions for the manager to check, not automatic orders: about 1 in 10 review flags turned out to be "
            "a real shortage in 2025, and the suggestion does not add safety stock or account for lead time."
        )
    st.divider()

    model_choice = st.radio(
        "KPI model", ["Seven-day item demand", "Seven-day shortage probability"], horizontal=True,
    )
    probability = model_choice == "Seven-day shortage probability"
    key = "shortage" if probability else "demand"
    model = shortage_model if probability else demand_model
    details = metadata[key]
    raw = details["raw_predictors"]
    met = details["metrics"]
    threshold = float(details.get("decision_threshold", 0.5))

    def fmt_pred(x: float) -> str:
        return f"{x:.1%}" if probability else f"{x:,.0f} units"

    # ------------------------------------------------------------------
    # 1. KPI definition, relevance, predictors and algorithm
    # ------------------------------------------------------------------
    st.subheader("KPI, predictors and model")
    if probability:
        st.markdown(
            "**KPI 2 — seven-day shortage probability:** the estimated chance that this item has any "
            "backorder during the next seven days.\n\n"
            "**Why it matters:** the inventory manager cannot review every item every day. Ranking items by "
            "shortage probability points attention, extra orders and supplier escalation where a stockout is "
            "most likely, protecting service continuity for patients without over-ordering everything."
        )
    else:
        st.markdown(
            "**KPI 1 — seven-day item demand:** the predicted number of units of this item that will be "
            "requested over the next seven days.\n\n"
            "**Why it matters:** this is how much stock the manager needs to cover the coming week. "
            "Under-estimating it leads to shortages and costly emergency purchases; over-estimating it ties "
            "up money and shelf space and risks expiry. It feeds order quantity and timing decisions."
        )
    with st.expander("Predictors and why they are relevant", expanded=False):
        reasons = PREDICTOR_REASONS[key]
        st.table(pd.DataFrame({"Predictor": list(reasons), "Why it is relevant": list(reasons.values())}))
    algorithm_note = (
        "**Algorithm: calibrated logistic regression.** The outcome is yes/no and rare, so a model that "
        "outputs a probability is the natural choice. Class weighting helps it learn from the few "
        "shortage examples, and sigmoid calibration then rescales the output so a stated 3% means "
        "roughly 3%. Logistic regression is also simple to explain to a non-technical manager."
        if probability else
        ("**Retained method: arrival-based formula** (seven-day arrival forecast × units used per patient). "
         "Week 11 evaluation compared it with four learned models on 2024 data only. The learned gradient-boosting "
         "models were no more accurate, and the Week 10 version predicted *lower* demand when more patients were "
         "expected on most rows, which made its what-if results misleading. A constrained version fixed the direction "
         "but was still less accurate. The formula is accurate, transparent and always moves in the right direction."
         if details.get("retained_type") == "formula" else
         "**Algorithm: gradient boosting regression.** Demand depends on combinations of factors (item "
         "scale, recent use, calendar and surges) that do not add up in a straight line. Gradient boosting "
         "captures these without a fixed formula, and Huber loss limits the pull of unusual surge days.")
    )
    st.markdown(algorithm_note)
    with st.expander("How the model was chosen (candidates compared on 2024 data only)"):
        sel = pd.DataFrame(details["selection"])
        if probability:
            sel = sel.rename(columns={"candidate": "Candidate", "validation_roc_auc": "Validation ROC-AUC",
                                      "validation_average_precision": "Validation average precision"})
        else:
            sel = sel.drop(columns=["fold_wape"], errors="ignore").rename(
                columns={"candidate": "Candidate", "mean_validation_wape": "Mean validation WAPE (4 rolling folds)",
                         "direction_violation_share": "Rows where +10% patients LOWERS demand"})
            sel["Mean validation WAPE (4 rolling folds)"] = sel["Mean validation WAPE (4 rolling folds)"].map("{:.2%}".format)
            if "Rows where +10% patients LOWERS demand" in sel:
                sel["Rows where +10% patients LOWERS demand"] = sel["Rows where +10% patients LOWERS demand"].map("{:.0%}".format)
        st.dataframe(sel, hide_index=True, width="stretch")
        st.caption(f"Rule: {details['selection_rule']}. Chosen: {details['algorithm']}.")
    with st.expander("Data availability and leakage checks"):
        st.markdown(
            "The prediction is made **after the daily inventory close on day t**. The outcome covers days "
            "t+1 through t+7. Therefore, only information known by the end of day t is permitted."
        )
        for check in metadata.get("leakage_checks", []):
            st.markdown(f"- {check}")
        st.caption(metadata.get("final_test_policy", ""))

    # ------------------------------------------------------------------
    # 2. Validation on the 2025 test period
    # ------------------------------------------------------------------
    st.subheader("How well it predicts (2025 test period, never used for training)")
    if probability:
        cols = st.columns(5)
        cols[0].metric("ROC-AUC", f"{met['roc_auc']:.3f}", help="Chance that a random shortage case is ranked above a random non-shortage case.")
        cols[1].metric("Average precision", f"{met['average_precision']:.3f}",
                       help=f"Area under the precision-recall curve. A random ranking would score about {met['observed_rate']:.3f}.")
        cols[2].metric("Shortages caught", f"{met['recall']:.0%}", help=f"Recall at the {threshold:.1%} screening threshold.")
        cols[3].metric("Alerts that were real", f"{met['precision']:.1%}", help=f"Precision at the {threshold:.1%} screening threshold.")
        cols[4].metric("Item-days flagged", f"{met['alert_share']:.1%}")
        st.warning(
            f"Shortages are rare: {met['positive_cases']} of {met['test_rows']:,} test item-days "
            f"({met['observed_rate']:.2%}). At the {threshold:.1%} screening threshold (chosen on 2024 data), "
            f"the model flags {met['alert_share']:.1%} of item-days and catches {met['recall']:.0%} of shortages, "
            f"but most alerts are false alarms. Its average predicted probability ({met['mean_predicted_probability']:.1%}) "
            f"is above the 2025 rate because shortages were more common in the 2024 training year. "
            "Use it to rank items for review, not as an automatic ordering rule."
        )
        per10 = met["precision"] * 10
        st.info(
            f"**In plain terms:** the model flags about {met['alert_share']:.0%} of item-days. Roughly "
            f"{per10:.0f} in 10 flagged items actually run short within a week, and it catches about "
            f"{met['recall'] * 10:.0f} in 10 of the shortages that happen. A flag means *check this item*, not *order now*."
        )
    else:
        cols = st.columns(4)
        cols[0].metric("Average miss (% of weekly demand)", f"{met['wape']:.1%}", help="WAPE: total absolute error ÷ total actual demand.")
        cols[1].metric("Typical miss per item-week", f"{met['mae_units']:,.0f} units", help="MAE (mean absolute error).")
        cols[2].metric("Large-miss measure", f"{met['rmse_units']:,.0f} units", help="RMSE: weights large misses more heavily.")
        cols[3].metric("R²", f"{met['r2']:.3f}", help="High mainly because items differ greatly in scale.")
        st.info(
            f"**In plain terms:** over 2025, the weekly demand estimate was off by about {met['wape']:.1%} of what was "
            f"actually used — roughly {met['mae_units']:,.0f} units per item per week. No learned model did better. "
            "Because the synthetic data was generated from arrivals × use rate, this accuracy is optimistic; on real "
            "hospital data the same method would need to be re-checked."
        )
    st.caption(
        f"Training: {metadata['training_period'][0]} to {metadata['training_period'][1]} · "
        f"Test: {metadata['test_period'][0]} to {metadata['test_period'][1]} · Split: chronological."
    )

    # ------------------------------------------------------------------
    # Original/revised comparison and error conditions
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Evaluation details")
    st.caption("Expand a section for the full tables. The key results are summarized above.")
    comp_box = st.expander("Baseline, original and revised comparison", expanded=False)
    comparison = pd.DataFrame(details.get("original_vs_revised", []))
    if not comparison.empty:
        comparison = comparison.rename(columns={
            "version": "Version", "model": "Method", "validation_wape": "Validation WAPE",
            "test_wape": "Test WAPE", "test_mae_units": "Test MAE (units)",
            "validation_average_precision": "Validation average precision",
            "test_roc_auc": "Test ROC-AUC", "test_average_precision": "Test average precision",
            "test_brier_score": "Test Brier score", "test_precision": "Test precision",
            "test_recall": "Test recall", "retained": "Retained",
            "direction_violation_share": "Rows where +10% patients lowers demand (validation)",
        })
        for col in ["Validation WAPE", "Test WAPE", "Rows where +10% patients lowers demand (validation)",
                    "Test precision", "Test recall"]:
            if col in comparison:
                comparison[col] = comparison[col].map(lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else v)
        comparison["Retained"] = comparison["Retained"].map({True: "✅ Retained", False: ""})
        comp_box.dataframe(comparison, hide_index=True, width="stretch")
        comp_box.caption("Validation = chosen on 2024 data only. Test = 2025, reported once after the choice was fixed.")
    st.info(details.get("improvement_conclusion", ""))

    st.markdown("**Where performance is weaker**")
    seg_box = st.expander("Full error table by item and condition", expanded=False)
    segments = error_analysis[error_analysis["kpi"] == ("Seven-day shortage" if probability else "Seven-day demand")].copy()
    if probability:
        item_segments = segments[segments["segment_type"] == "item"].sort_values(
            ["recall", "false_positives"], ascending=[True, False]
        )
        display_cols = ["segment", "rows", "positives", "average_precision", "precision", "recall",
                        "false_positives", "false_negatives"]
        seg_box.dataframe(item_segments[display_cols].rename(columns={
            "segment": "Item", "rows": "Rows", "positives": "Shortages",
            "average_precision": "Average precision", "precision": "Precision", "recall": "Recall",
            "false_positives": "False alerts", "false_negatives": "Missed shortages",
        }), hide_index=True, width="stretch")
        pos = item_segments[pd.to_numeric(item_segments["positives"], errors="coerce") > 0]
        names = ", ".join(ITEM_NAME.get(i, i) for i in pos["segment"])
        weakest = pos.sort_values("recall").iloc[0] if not pos.empty else None
        st.warning(
            f"Only {len(pos)} items ({names}) had shortages in the 2025 test period"
            + (f"; the weakest recall is {ITEM_NAME.get(weakest['segment'], weakest['segment'])} "
               f"({float(weakest['recall']):.0%})" if weakest is not None else "")
            + ". Items with no observed shortages still generate false alerts. Subgroup estimates are unstable "
            "because the positive counts are very small. Treat the score as a review queue, not an order instruction."
        )
    else:
        weak = segments.sort_values("wape", ascending=False).head(10).copy()
        def segment_label(row: pd.Series) -> str:
            if row["segment_type"] == "surge":
                return "Surge days" if str(row["segment"]) == "1" else "Normal days"
            if row["segment_type"] == "item":
                return ITEM_NAME.get(row["segment"], row["segment"])
            return f"{str(row['segment']).capitalize()} demand weeks"
        weak["Segment"] = weak.apply(segment_label, axis=1)
        weak["Average miss"] = pd.to_numeric(weak["wape"]).map("{:.1%}".format)
        weak["Typical miss (units)"] = pd.to_numeric(weak["mae"]).round(0).astype(int)
        weak["Average bias (units)"] = pd.to_numeric(weak["bias"]).map("{:+,.0f}".format)
        seg_box.dataframe(weak[["segment_type", "Segment", "rows", "Average miss", "Typical miss (units)", "Average bias (units)"]]
                     .rename(columns={"segment_type": "Segment type", "rows": "Item-weeks"}),
                     hide_index=True, width="stretch")
        seg_box.caption("Bias: negative = the estimate was too low (risk of running short); positive = too high.")
        surge = segments[segments["segment_type"] == "surge"].set_index("segment")
        band = segments[segments["segment_type"] == "demand band"].set_index("segment")
        s_on = float(surge.loc["1", "wape"]) if "1" in surge.index else float("nan")
        s_off = float(surge.loc["0", "wape"]) if "0" in surge.index else float("nan")
        surge_bias = float(surge.loc["1", "bias"]) if "1" in surge.index else float("nan")
        st.warning(
            f"Errors are about {s_on / s_off:.0f}× larger during surge periods ({s_on:.1%} average miss vs. {s_off:.1%} "
            f"otherwise), and during surges the estimate is {'too low' if surge_bias < 0 else 'too high'} by about "
            f"{abs(surge_bias):,.0f} units per item-week on average, because the arrival forecast does not see a surge "
            "coming. During a surge, hold extra safety stock and use the what-if to test a higher arrival forecast "
            "rather than rely on the baseline estimate."
        )

    # ------------------------------------------------------------------
    # 3a/3b. Overall predictor importance + contextual interpretation
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Overall predictor importance (whole test period)")
    model_name = "Seven-day shortage" if probability else "Seven-day demand"
    imp = importance[importance["model"] == model_name].copy()
    imp["Predictor"] = imp["predictor"].map(lambda c: PREDICTOR_LABELS.get(c, readable(c)))
    fig_imp = px.bar(
        imp.sort_values("importance"), x="importance", y="Predictor", orientation="h", error_x="std",
        labels={"importance": ("Drop in ROC-AUC when shuffled" if probability
                               else "Increase in average error (units) when shuffled"), "Predictor": ""},
    )
    fig_imp.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig_imp, width="stretch")
    if probability:
        st.markdown(
            "**What this means:** the model leans most on **closing stock** compared with recent demand: "
            "low physical stock against a busy week is the clearest warning sign. **Recent demand** (yesterday "
            "and the seven-day average) comes next, because a rising trend uses stock up faster. Surges, lead "
            "time and the arrival forecast add smaller signals. **Days of supply** ranks low, not because "
            "coverage is unimportant but because closing stock and recent demand already carry the same "
            "information. Supplier events are rare, so they matter on the few days they happen."
        )
    else:
        st.markdown(
            "**What this means:** **units used per arrival** dominates because it sets each item's scale "
            "(gloves vs. specimen cups); shuffling it between items creates huge errors. The **seven-day arrival "
            "forecast** ranks lower only because it varies little from week to week (about ±15%). Within one item, "
            "it drives the prediction one-for-one: 10% more expected patients means 10% more expected demand."
        )
    st.caption(
        "Method: permutation importance. Each predictor is shuffled in turn across the 2025 test data and "
        "the loss of accuracy is measured (bars show ± one standard deviation over 8 shuffles). It shows "
        "what the model relies on overall. It does not show cause and effect."
    )

    # ------------------------------------------------------------------
    # Baseline prediction for one item and date
    # ------------------------------------------------------------------
    eligible = model_data[model_data["item_id"].isin(selected_items)].copy()
    eligible = eligible[(eligible["date"].dt.date >= START) & (eligible["date"].dt.date <= END)]
    if eligible.empty:
        stop_with("No model rows match the current sidebar filters. Model rows end on 2025-12-24, "
                  "because each needs seven days of known outcomes.")

    st.divider()
    st.subheader("Individual prediction")
    left, right = st.columns(2)
    with left:
        model_item = st.selectbox(
            "Item", [i for i in ITEMS["item_id"] if i in set(eligible["item_id"])],
            format_func=lambda i: ITEM_NAME.get(i, i), key=f"model_item_{key}",
        )
    item_rows = eligible[eligible["item_id"] == model_item]
    valid_dates = sorted(item_rows["date"].dt.date.unique())
    with right:
        model_date = st.selectbox("Date", valid_dates, index=len(valid_dates) - 1, key=f"model_date_{key}",
                                  format_func=lambda d: d.strftime("%Y-%m-%d"))
    baseline_row = item_rows[item_rows["date"].dt.date == model_date].tail(1).reset_index(drop=True)
    baseline_prediction = model_prediction(model, baseline_row[raw], details, probability)
    actual_col = "target_shortage_next_7d" if probability else "target_demand_next_7d"
    actual = baseline_row[actual_col].iloc[0]

    m1, m2 = st.columns(2)
    m1.metric("Baseline model prediction", fmt_pred(baseline_prediction))
    m2.metric("What actually happened (known in hindsight)",
              ("A shortage occurred" if actual == 1 else "No shortage") if probability else f"{actual:,.0f} units")
    st.markdown(f"**Contributions to this baseline prediction** ({ITEM_NAME.get(model_item, model_item)}, {model_date:%Y-%m-%d})")
    base_contrib = reference_replacement_contributions(model, baseline_row, details, probability)
    st.plotly_chart(contribution_chart(base_contrib, probability), width="stretch", key=f"base_contrib_{key}")
    st.markdown(contribution_sentence(base_contrib, probability))
    st.caption(
        "Method: each predictor in turn is set to its typical 2024 value (median, or most common for "
        "supplier event) and the prediction is recomputed. The difference is that predictor's approximate "
        "contribution to *this* prediction. Interactions mean the bars need not add up exactly. These are "
        "model sensitivities, not causal effects."
    )

    # ------------------------------------------------------------------
    # 3c. What-if analysis
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("What-if scenario")
    st.caption(
        f"Baseline = the recorded values for {ITEM_NAME.get(model_item, model_item)} on {model_date:%Y-%m-%d} "
        f"(prediction {fmt_pred(baseline_prediction)}). Change the permitted scenario inputs and run it. The result is a "
        "model-based scenario, not a guaranteed effect of a real-world change."
    )
    if probability:
        control_table = pd.DataFrame([
            ["Planning inputs", "Closing stock; supplier lead time", "The manager can influence these through ordering, stock transfers or supplier choice, though not instantly."],
            ["External conditions", "Seven-day arrival forecast; surge; supplier event", "Outside the manager's control, but worth testing as scenarios."],
            ["Fixed / observed", "Recent demand; 28-day average arrivals", "Already happened for the selected date, so locked."],
            ["Derived", "Days of supply", "Recalculated from closing stock; cannot be edited on its own."],
        ], columns=["Type", "Predictors", "How to interpret"])
    else:
        control_table = pd.DataFrame([
            ["External conditions", "Seven-day arrival forecast", "Outside the manager's control, but the main thing worth testing (e.g. a flu wave or a quiet week)."],
            ["Fixed item characteristic", "Units used per arrival", "An item assumption; locked in the what-if."],
            ["Fixed / observed (consistency check only)", "One-day arrival forecast; current arrivals", "Shown to check that the seven-day scenario is believable."],
            ["Decision output", "Seven-day demand", "Feeds the order quantity; not an input."],
        ], columns=["Type", "Predictors", "How to interpret"])
    with st.expander("Which predictors can change?"):
        st.dataframe(control_table, hide_index=True, width="stretch")
    ratio_lo, ratio_hi = details.get("forecast_ratio_limits", list(MF.FORECAST_RATIO_LIMITS))
    with st.expander("How input checks work"):
        st.markdown(
            "- **Plausible ranges:** each editable number is limited to values seen in the 2024 development data. "
            "Closing stock uses **this item's own** range (for example, specimen cups never exceeded 700 units).\n"
            "- **Locked fields:** values that already happened for the selected date (recent demand, current arrivals, "
            "holiday) and fixed item characteristics cannot be edited.\n"
            "- **Related variables stay consistent:** days of supply is recalculated when closing stock changes.\n"
            f"- **Cross-check:** the seven-day arrival forecast must stay between {ratio_lo:.2f}× and {ratio_hi:.2f}× "
            "seven times the one-day forecast (2024 values were always 0.83–1.25×). A scenario outside this band "
            "is rejected with an explanation and **no prediction is shown**, rather than producing a number from an "
            "impossible situation."
        )

    values = baseline_row[raw].iloc[0].to_dict()
    context = baseline_row.iloc[0].to_dict()   # full row: fixed fields used for display and checks
    observed_ranges = dict(details.get("input_ranges", {}))
    observed_ranges.update(details.get("input_ranges_by_item", {}).get(model_item, {}))

    def float_limits(feature: str) -> tuple[float, float]:
        baseline = float(context[feature])
        limits = observed_ranges.get(feature, {"min": 0.0, "max": max(1.0, baseline * 2)})
        return min(float(limits["min"]), baseline), max(float(limits["max"]), baseline)

    def int_limits(feature: str) -> tuple[int, int]:
        low, high = float_limits(feature)
        return int(np.floor(low)), int(np.ceil(high))

    with st.form(f"what_if_{key}"):
        c1, c2, c3 = st.columns(3)
        if probability:
            with c1:
                lo, hi = int_limits("closing_on_hand")
                values["closing_on_hand"] = st.number_input(
                    "Closing stock (planning input)", min_value=max(0, lo), max_value=hi,
                    value=int(values["closing_on_hand"]), step=10,
                    help=f"Limited to this item's 2024 range: {lo:,}–{hi:,} units.",
                )
                lo, hi = int_limits("lead_time_days")
                values["lead_time_days"] = st.number_input(
                    "Supplier lead time, days (planning input)", min_value=max(1, lo), max_value=hi,
                    value=int(values["lead_time_days"]), step=1,
                    help="Limited to lead times used by the project's suppliers (2–5 days).",
                )
                st.text_input("Days of supply (derived)", value="Recalculated from closing stock", disabled=True)
            with c2:
                lo, hi = float_limits("forecast_arrivals_7d")
                values["forecast_arrivals_7d"] = st.number_input(
                    "Seven-day arrival forecast (external)", min_value=max(0.0, lo), max_value=hi,
                    value=float(values["forecast_arrivals_7d"]), step=25.0,
                    help=f"Limited to the 2024 range: {lo:,.0f}–{hi:,.0f} patients.",
                )
                st.text_input("Previous-day demand (fixed)", value=fmt_value(values["demand_lag1"]), disabled=True)
                st.text_input("Trailing seven-day demand (fixed)", value=fmt_value(values["demand_roll7"]), disabled=True)
            with c3:
                values["surge_flag"] = int(st.checkbox("Demand surge (external)", value=bool(values["surge_flag"])))
                events = MF.SUPPLIER_EVENTS
                current = str(values["supplier_event"])
                values["supplier_event"] = st.selectbox("Supplier event (external)", events,
                                                        index=events.index(current) if current in events else 0)
        else:
            with c1:
                lo, hi = float_limits("forecast_arrivals_7d")
                values["forecast_arrivals_7d"] = st.number_input(
                    "Seven-day arrival forecast (external)", min_value=max(0.0, lo), max_value=hi,
                    value=float(values["forecast_arrivals_7d"]), step=25.0,
                    help=f"Limited to the 2024 range: {lo:,.0f}–{hi:,.0f} patients.",
                )
            with c2:
                st.text_input("One-day arrival forecast (fixed, for the cross-check)",
                              value=fmt_value(context["forecast_arrivals_1d"]), disabled=True)
                st.text_input("Current arrivals (fixed)", value=fmt_value(context["ed_arrivals"]), disabled=True)
            with c3:
                st.text_input("Units used per arrival (fixed item characteristic)",
                              value=fmt_value(values["use_rate_per_arrival"]), disabled=True)
        run = st.form_submit_button("Run model-based scenario", type="primary")

    if not run:
        st.caption("Run the scenario to see the side-by-side comparison, its contributions, and an automatic summary.")
        return

    # Keep related variables consistent: days of supply follows closing stock,
    # using the selected row's recent average daily demand.
    if probability:
        stock_change = float(values["closing_on_hand"]) - float(baseline_row["closing_on_hand"].iloc[0])
        values["days_of_supply"] = max(
            0.0,
            float(baseline_row["days_of_supply"].iloc[0]) + stock_change / max(float(values["demand_roll7"]), 1.0),
        )
    invalid = []
    ratio = MF.forecast_ratio(values["forecast_arrivals_7d"], context["forecast_arrivals_1d"])
    if not ratio_lo <= ratio <= ratio_hi:
        invalid.append(
            f"The seven-day arrival forecast ({values['forecast_arrivals_7d']:,.0f}) is {ratio:.2f}× seven times "
            f"tomorrow's forecast ({float(context['forecast_arrivals_1d']):,.0f} × 7 = "
            f"{7 * float(context['forecast_arrivals_1d']):,.0f}). That is outside the plausible band "
            f"({ratio_lo:.2f}–{ratio_hi:.2f}×), so the week would be inconsistent with tomorrow's forecast."
        )
    if invalid:
        for message in invalid:
            st.error(message)
        st.info("No scenario prediction was produced, because a number from an inconsistent scenario would be "
                "misleading. Adjust the inputs and run the scenario again.")
        return

    updated = pd.DataFrame([values])
    updated_prediction = model_prediction(model, updated, details, probability)
    delta = updated_prediction - baseline_prediction

    # ------------------------------------------------------------------
    # 3e. Side-by-side comparison
    # ------------------------------------------------------------------
    st.markdown("**Side-by-side comparison**")
    comparison_rows = []
    for feature in raw:
        before = baseline_row[feature].iloc[0]
        after = updated[feature].iloc[0]
        if fmt_value(before) != fmt_value(after):
            comparison_rows.append({
                "": PREDICTOR_LABELS.get(feature, readable(feature)),
                "Baseline": fmt_value(before), "Scenario": fmt_value(after),
                "Difference": (f"{float(after) - float(before):+,.2f}"
                               if isinstance(before, (int, float, np.integer, np.floating)) else "Changed"),
            })
    comparison_rows.append({
        "": "▶ Predicted " + ("shortage probability" if probability else "seven-day demand"),
        "Baseline": fmt_pred(baseline_prediction), "Scenario": fmt_pred(updated_prediction),
        "Difference": f"{delta * 100:+.1f} pts" if probability else f"{delta:+,.0f} units",
    })
    if probability:
        comparison_rows.append({
            "": "▶ Flagged for review?", "Baseline": "Yes" if baseline_prediction >= threshold else "No",
            "Scenario": "Yes" if updated_prediction >= threshold else "No", "Difference": "",
        })
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
    if len(comparison_rows) <= (2 if probability else 1):
        st.caption("No inputs were changed, so the scenario repeats the baseline.")

    contrib = reference_replacement_contributions(model, updated, details, probability)
    st.markdown("**Contributions to the scenario prediction**")
    st.plotly_chart(contribution_chart(contrib, probability), width="stretch", key=f"scen_contrib_{key}")

    # ------------------------------------------------------------------
    # 3f. Automatically generated summary
    # ------------------------------------------------------------------
    changed = [f"{r['']} changed from {r['Baseline']} to {r['Scenario']}" for r in comparison_rows
               if not r[""].startswith("▶")]
    input_text = "; ".join(changed) + "." if changed else "No inputs were changed, so this repeats the baseline."
    if probability:
        result_text = (f"The estimated seven-day shortage probability moved from {baseline_prediction:.1%} to "
                       f"{updated_prediction:.1%} ({delta * 100:+.1f} percentage points).")
        crossed = (baseline_prediction >= threshold) != (updated_prediction >= threshold)
        if updated_prediction >= threshold:
            implication = (f"This is at or above the {threshold:.1%} screening threshold"
                           + (", so the scenario moves the item **onto** the review list" if crossed else ", so the item stays flagged")
                           + ": check stock, open orders and the supplier before deciding whether to order early or escalate.")
        else:
            implication = (f"This is below the {threshold:.1%} screening threshold"
                           + (", so the scenario takes the item **off** the review list" if crossed else ", so the item is not flagged")
                           + ". Routine review still applies.")
        tradeoff = (
            "Acting early (an extra order, a transfer or a supplier call) lowers the risk of running out, but costs "
            "staff time, cash and shelf space, and most flags are false alarms: roughly "
            f"{met['precision'] * 10:.0f} in 10 flagged items actually ran short in 2025. Not acting saves that effort "
            f"but the model misses about {(1 - met['recall']) * 10:.0f} in 10 real shortages, so a low score is not a guarantee."
        )
    else:
        pct = delta / baseline_prediction if baseline_prediction else 0.0
        result_text = (f"Predicted seven-day demand moved from {baseline_prediction:,.0f} to {updated_prediction:,.0f} "
                       f"units ({delta:+,.0f} units, {pct:+.1%}).")
        implication = (("If this scenario happened, the week's stock target would need about "
                        f"{abs(delta):,.0f} {'more' if delta > 0 else 'fewer'} units")
                       if abs(delta) >= 1 else "The scenario does not materially change the week's stock need")
        implication += "; the final order should still account for stock on hand, open orders, lead time and case packs."
        tradeoff = (
            "Ordering to the higher figure protects against running out if patient volume really rises, but ties up "
            "cash and shelf space and risks expiry if it does not. Ordering to the lower figure saves money but leaves "
            f"less buffer. The estimate typically misses by about {met['wape']:.0%} of a week's demand, and by more "
            "during surges, so safety stock is still needed."
        )
    st.subheader("Automatically generated scenario summary")
    st.success(
        f"**Scenario inputs.** {input_text}\n\n"
        f"**Comparison.** {result_text}\n\n"
        f"**What drives it.** {contribution_sentence(contrib, probability)}\n\n"
        f"**Implication.** {implication}\n\n"
        f"**Trade-offs.** {tradeoff}\n\n"
        "**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model "
        "predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. "
        "Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can "
        "still differ from real operations. Not an automatic purchasing or clinical decision."
    )


# ---------------------------------------------------------------------------
# Page 6 — Data & Assumptions
# ---------------------------------------------------------------------------
def page_data() -> None:
    st.title("Data & Assumptions")
    st.caption("Can I trust this?")

    st.subheader("Item assumptions")
    st.dataframe(ITEMS.rename(columns=readable), hide_index=True, width="stretch")

    st.subheader("Validation results")
    n_pass = int((VALIDATION["status"].str.upper() == "PASS").sum())
    n_fail = int((VALIDATION["status"].str.upper() == "FAIL").sum())
    n_total = len(VALIDATION)
    m = st.columns(3)
    m[0].metric("PASS", f"{n_pass} of {n_total}")
    m[1].metric("FAIL", f"{n_fail}")
    m[2].metric("Checks", f"{n_total}")
    if n_pass == n_total:
        st.success(f"{n_pass} of {n_total} validation checks PASS.")
    else:
        st.error(f"Only {n_pass} of {n_total} validation checks PASS. Review the FAIL rows below.")
    st.dataframe(VALIDATION.rename(columns=readable), hide_index=True, width="stretch")

    st.subheader("About the data")
    st.markdown(
        f"""
**Source.** All data is **synthetic**: generated by a script with a fixed random seed (20261009),
not taken from real hospital records. Item assumptions are manually constructed and illustrative.

**Coverage.** {DATA_MIN:%Y-%m-%d} to {DATA_MAX:%Y-%m-%d}
({ARRIVALS['date'].nunique():,} days), {ITEMS['item_id'].nunique()} routine ED consumables.

**Row counts.**
- `generated_ed_arrivals.csv`: {len(ARRIVALS):,} rows
- `item_assumptions.csv`: {len(ITEMS):,} rows
- `inventory_decision_daily.csv`: {len(DAILY):,} rows × {DAILY.shape[1]} columns
- `scenario_comparison.csv`: {len(SCENARIO):,} rows
- `validation_results.csv`: {len(VALIDATION):,} rows

**v1.1 simulation assumptions.**
- Arrival surges: multi-day demand surge episodes and one-day spikes (`surge_flag` = 1).
- Ordering on review days, Monday and Thursday, or on any day the risk score is ≥ 0.50.
- Supplier disruptions: each order has an 8% chance of arriving 2–6 days late and a 5% chance of
  being short-shipped (30–70% of the units ordered).

**Predictive extension.** Version 3 evaluates a seven-day demand estimate and a seven-day shortage model on a
chronological 2025 holdout period. For demand, the arrival-based formula was retained because no learned model
beat it on 2024 validation data. Both are intended for demonstration and decision screening only.

**Limitations.** {FOOTER_TEXT}
"""
    )

    st.subheader("Download")
    export = DAILY_F.drop(columns=["item_name", "category"]).copy()
    export.insert(2, "item_name", DAILY_F["item_name"])
    export.insert(3, "category", DAILY_F["category"])
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download filtered daily table (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"inventory_decision_daily_{START:%Y%m%d}_{END:%Y%m%d}.csv",
        mime="text/csv",
        disabled=export.empty,
        help="Rows of inventory_decision_daily.csv matching the sidebar date and item filters, "
             "with item name and category added.",
    )
    st.caption(f"{len(export):,} rows match the current filters.")


# ---------------------------------------------------------------------------
# Navigation + footer
# ---------------------------------------------------------------------------
pages = [
    st.Page(page_overview, title="Overview", icon="🚨", url_path="overview", default=True),
    st.Page(page_item_explorer, title="Item Explorer", icon="🔍", url_path="item-explorer"),
    st.Page(page_trends, title="Trends & Mix", icon="📈", url_path="trends"),
    st.Page(page_scenarios, title="Scenario Comparison", icon="⚖️", url_path="scenarios"),
    st.Page(page_predictive_models, title="Predictive Models", icon="🤖", url_path="predictive-models"),
    st.Page(page_data, title="Data & Assumptions", icon="🗂️", url_path="data"),
]
st.navigation(pages).run()
footer()
