"""
ED Supply Decision Support — Version 1 (Descriptive Dashboard)

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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent          # app_build/ (next to data/)
DATA_DIR = APP_ROOT / "data"

PATHS = {
    "arrivals": DATA_DIR / "raw" / "generated_ed_arrivals.csv",
    "items": DATA_DIR / "raw" / "item_assumptions.csv",
    "daily": DATA_DIR / "final" / "inventory_decision_daily.csv",
    "scenario": DATA_DIR / "final" / "scenario_comparison.csv",
    "validation": DATA_DIR / "final" / "validation_results.csv",
}

ASSUMPTION_VERSION = "v1.1-2026-10-19"
FOOTER_TEXT = (
    "Prototype using synthetic data and illustrative assumptions "
    f"(assumption version {ASSUMPTION_VERSION}). Not for real ordering or "
    "clinical decisions. The risk score is a ranking proxy, not a probability."
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
# Page 5 — Data & Assumptions
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

**Limitations.** {FOOTER_TEXT} Forecast columns are shown as stored; Version 1 does not train or run
any forecasting model.
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
    st.Page(page_data, title="Data & Assumptions", icon="🗂️", url_path="data"),
]
st.navigation(pages).run()
footer()
