# ED Supply Decision Support — Version 1 (Descriptive Dashboard)

Streamlit app for exploring two years (2024-01-01 to 2025-12-31) of **synthetic** emergency department
arrivals and supply data for 8 routine consumables. It shows KPIs, charts and tables that respond to the
sidebar filters, and a risk queue showing which items need attention.

Data version: `v1.1-2026-10-19`. Specification: `docs/app_specification.md`.
Team: Monica Hall, Abubakar Sani (MS Analytics, Saint Louis University).

## Run it

From the `app_build/` folder (the one that contains `app.py` and `data/`):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+ is required. The app reads these files (read-only) and does not write to them:

```
data/raw/generated_ed_arrivals.csv
data/raw/item_assumptions.csv
data/final/inventory_decision_daily.csv
data/final/scenario_comparison.csv
data/final/validation_results.csv
```

## Pages

1. **Overview**: 8 KPIs, daily arrivals with 7-day average, holidays and surge shading, arrivals by weekday, and the risk queue for an as-of date.
2. **Item Explorer**: per-item KPIs, an explanation of the item's status on the as-of date, and inventory, demand/receipts, risk and days-of-supply charts.
3. **Trends & Mix**: monthly demand by category, order spend by item, high-risk days by month, arrivals by day type, and a per-item summary table.
4. **Scenario Comparison**: low/base/high demand scenarios for one item and date.
5. **Data & Assumptions**: item assumptions, validation results (13/13 PASS), data notes, and a CSV download of the filtered daily table.

All calculation rules (R1–R8 in the spec) are implemented once, in the "Calculation rules" section of
`app.py`, and every page uses those functions.

## Limitations

Prototype using synthetic data and illustrative assumptions (assumption version v1.1-2026-10-19).
Not for real ordering or clinical decisions. The risk score is a ranking proxy, not a probability.
Version 1 is descriptive only: it does not train or run any forecasting model.
