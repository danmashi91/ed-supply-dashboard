# App Specification — ED Supply Decision Support (Version 1: Descriptive Dashboard)

Project: Patient Admission Driven Demand Forecasting and Inventory Decision Support
Team: Monica Hall, Abubakar Sani (MS Analytics, Saint Louis University)

## 1. What to build

Build a Python **Streamlit** app that lets an emergency department (ED) inventory manager explore two years of daily ED arrivals and supply data for 8 routine consumables, and see which items need attention.

Version 1 is **descriptive only**. It reads existing CSV files and shows KPIs, charts, and tables that change when the user moves filters. It does **not** train or run any forecasting model. The forecast columns already in the data are displayed as-is.

The data is synthetic (generated, not real hospital records). The app must say this clearly on every page.

## 2. Tech requirements

- Python 3.10+, `streamlit`, `pandas`, `plotly` (plotly.express). No other dependencies.
- Single multipage app: `app.py` plus `pages/` folder (Streamlit's built-in multipage), or one `app.py` using `st.navigation`. Either is fine.
- Load every CSV once with `@st.cache_data`. Parse `date` columns as dates.
- Paths are relative to the project root (the app sits next to the `data/` folder).
- Read-only. The app never writes to or changes the data files.
- Include `requirements.txt` and a short `README` with: `pip install -r requirements.txt` then `streamlit run app.py`.

## 3. Data files and schema

### 3.1 `data/raw/generated_ed_arrivals.csv` — one row per date (731 rows)
| Column | Type | Meaning |
|---|---|---|
| date | date (PK) | 2024-01-01 to 2025-12-31 |
| ed_arrivals | int | ED arrivals that day |
| expected_arrivals_parameter | float | Expected value used to generate arrivals (not shown in the app) |
| day_of_week | string | Monday … Sunday |
| weekend_flag | int 0/1 | 1 = Saturday or Sunday |
| holiday_flag | int 0/1 | 1 = selected holiday |
| source_status | string | Always "generated" |

### 3.2 `data/raw/item_assumptions.csv` — one row per item (8 rows)
| Column | Type | Meaning |
|---|---|---|
| item_id | string (PK) | IVKIT, GAUZE, GLOVEM, SPECCUP, SALFLUSH, MASK, UNDERPAD, BLOODTUBE |
| item_name | string | Display name, e.g. "IV start kit" |
| category | string | vascular access, wound care, PPE, laboratory, patient care |
| use_rate | float | Assumed units used per ED arrival |
| use_cv | float | Assumed variability of use |
| lead_time_days | int | Supplier lead time |
| case_pack | int | Order multiple |
| unit_cost | float | Cost per unit (USD, illustrative) |
| service_level_target | float | Target availability, e.g. 0.97 |
| safety_stock_days | int | Safety stock in days of demand |
| assumption_version | string | e.g. v1.0-2026-10-09 |
| source_status | string | "manually_constructed_assumption" |

### 3.3 `data/final/inventory_decision_daily.csv` — one row per date × item (5,848 rows, PK = date + item_id)
| Column | Type | Meaning |
|---|---|---|
| date | date | Day |
| item_id | string (FK → item_assumptions) | Item |
| ed_arrivals | int | Same value for all 8 items on a date (**repeated per item — see rule R1**) |
| forecast_arrivals_1d | float | One-day-ahead arrival forecast (same for all items on a date) |
| forecast_arrivals_7d | float | Sum of next 7 days' arrival forecast (same for all items on a date) |
| item_demand_requested | int | Units requested |
| item_issued_qty | int | Units issued (≤ requested) |
| backorder_qty | int | Unfilled units carried forward |
| opening_on_hand | int | Stock at start of day |
| receipts_qty | int | Units received that day |
| closing_on_hand | int | opening + receipts − issued |
| open_order_qty | int | Units ordered, not yet received |
| inventory_position | int | closing_on_hand + open_order_qty − backorder_qty |
| forecast_item_demand_7d | float | 7-day item demand forecast |
| days_of_supply | float | inventory_position ÷ average daily forecast demand |
| stockout_risk_proxy | float 0–1 | Ranking score for shortage risk. **Not a probability.** |
| recommended_order_qty | int | Recommended order, rounded to case_pack |
| recommended_action | string | "escalate", "order", or "monitor" |
| lead_time_days | int | Copied from item assumptions |
| case_pack | int | Copied from item assumptions |
| unit_cost | float | Copied from item assumptions |
| estimated_order_cost | float | recommended_order_qty × unit_cost |
| use_rate_per_arrival | float | Copied from item assumptions |
| service_level_target | float | Copied from item assumptions |
| assumption_version | string | Assumption version |
| record_status | string | Provenance label |

### 3.4 `data/final/scenario_comparison.csv` — one row per date × item × scenario (17,544 rows, PK = date + item_id + scenario)
| Column | Type | Meaning |
|---|---|---|
| date | date | Day |
| item_id | string (FK) | Item |
| scenario | string | "low", "base", "high" (always show in this order) |
| demand_multiplier | float | 0.85, 1.00, 1.20 |
| forecast_item_demand_7d | float | 7-day item demand under the scenario |
| inventory_position | int | Same for all three scenarios (physical stock does not change) |
| days_of_supply | float | Under the scenario |
| stockout_risk_proxy | float 0–1 | Under the scenario |
| recommended_order_qty | int | Under the scenario |
| assumption_version | string | Assumption version |

### 3.5 `data/final/validation_results.csv` — 13 rows
Columns: `check_id, validation_check, status, observed, expected, action_taken`. Display only.

### 3.6 Joins
- Join item tables to `item_assumptions` on `item_id` to get `item_name` and `category`. Always display `item_name`, not the code.
- Join date-level views to `generated_ed_arrivals` on `date` for `day_of_week`, `weekend_flag`, `holiday_flag`.

## 4. Calculation rules (must follow exactly)

- **R1 — No double counting arrivals.** `ed_arrivals`, `forecast_arrivals_1d` and `forecast_arrivals_7d` repeat on every item row. For any arrival KPI or chart, use `generated_ed_arrivals.csv`, or de-duplicate the daily table to one row per date first. Item filters must **not** change arrival numbers.
- **R2 — Fill rate** = sum(item_issued_qty) ÷ sum(item_demand_requested) over the filtered rows. Show as a percentage.
- **R3 — Arrival forecast WAPE** = sum(|ed_arrivals − forecast_arrivals_1d|) ÷ sum(ed_arrivals) over the filtered dates, one row per date.
- **R4 — High-risk item-days** = count of rows where stockout_risk_proxy ≥ the risk threshold (sidebar, default 0.50).
- **R5 — Coverage below lead time** = days_of_supply < lead_time_days. Show it as a flag.
- **R6 — Risk queue sort order:** stockout_risk_proxy descending, then days_of_supply ascending.
- **R7 — Order cost** = sum(estimated_order_cost). Format as USD with no decimals.

## 5. Global layout

**Sidebar (all pages):**
- Date range picker. Default = full range 2024-01-01 to 2025-12-31. Min and max are locked to the data range.
- Category multiselect (default all).
- Item multiselect, limited to the chosen categories (default all).
- Risk threshold slider, 0.00 to 1.00, step 0.05, default 0.50.

**Footer banner on every page:** "Prototype using synthetic data and illustrative assumptions (assumption version v1.0-2026-10-09). Not for real ordering or clinical decisions. The risk score is a ranking proxy, not a probability."

If filters return no rows, show a friendly message instead of an error.

## 6. Pages

### Page 1 — Overview ("What needs attention?")
- **KPI row (6 tiles):** Total ED arrivals · Average daily arrivals · Fill rate (R2) · Arrival forecast WAPE (R3) · High-risk item-days (R4) · Total recommended order cost (R7).
- **Chart A:** Daily ED arrivals (line) with a 7-day rolling average (second line). Mark holidays with points.
- **Chart B:** Average arrivals by day of week (bar, Monday to Sunday order).
- **Risk queue table** for one **as-of date** (date picker on the page, default = last date in the selected range). One row per selected item: item name, category, inventory position, days of supply, lead time, coverage-below-lead-time flag (R5), risk score, recommended action, recommended order qty, estimated order cost. Sort by R6. Highlight rows where risk ≥ threshold.

### Page 2 — Item Explorer ("Why is this item flagged?")
- Single item selector (from the sidebar item list).
- **KPI row:** Current days of supply · Lead time · Average daily demand in range · Total units requested · Total order cost in range.
- **Chart C:** Inventory over time. Lines for closing_on_hand and inventory_position, plus a dashed horizontal reference line for safety stock = safety_stock_days × average daily demand in the range.
- **Chart D:** Daily demand requested vs issued (lines), with receipts as bars.
- **Chart E:** Days of supply over time, with a dashed line at lead_time_days.
- **Explanation box (plain text):** For the as-of date: "On {date}, {item_name} had {days_of_supply} days of supply against a {lead_time}-day lead time. Risk score {risk}. Recommended action: {action}, order {qty} units ({qty/case_pack} cases) costing ${cost}."

### Page 3 — Trends & Mix ("What drives demand and cost?")
- **Chart F:** Monthly units requested by category (stacked bar).
- **Chart G:** Total recommended order cost by item (horizontal bar, sorted).
- **Chart H:** Average daily arrivals: weekday vs weekend vs holiday (bar).
- **Table:** Per item for the filtered range: total requested, total issued, fill rate, average days of supply, number of escalate / order / monitor days, total order cost.

### Page 4 — Scenario Comparison ("What if demand is higher or lower?")
- Pickers: as-of date and one item.
- **Table:** 3 rows (low, base, high): demand multiplier, 7-day item demand, inventory position, days of supply, risk score, recommended order qty, estimated cost (= recommended_order_qty × unit_cost from item assumptions).
- **Chart I:** Grouped bar of recommended order qty and days of supply by scenario.
- Note under the chart: "Physical stock is the same in all scenarios. Only forecast demand changes."

### Page 5 — Data & Assumptions ("Can I trust this?")
- Item assumptions table (all columns, readable headers).
- Validation results table, with a PASS/FAIL count summary (expected: 13 of 13 PASS).
- Short text: data source (synthetic, fixed seed 20261009), coverage dates, row counts, and the limitations in the footer banner.
- Download button for the currently filtered daily table as CSV.

## 7. Out of scope for Version 1
- Training, refitting or running any forecasting or ML model (comes in Version 2).
- Approving or overriding orders, an action log, or writing any data.
- Login, user accounts, a database server, or ERP/supplier connections.
- Medication or patient-level data.

## 8. Acceptance checks (with no filters, full date range)
The app is correct if it shows these values:
- Total ED arrivals = **138,611**; average daily arrivals ≈ **189.6**
- Fill rate = **100.0%**
- Arrival forecast WAPE = **7.64%**
- High-risk item-days at threshold 0.50 = **8** (all on 2024-01-01)
- Total recommended order cost ≈ **$582,580**
- Validation: **13 of 13 PASS**
- Changing the item filter does **not** change total ED arrivals or WAPE (R1).
- The risk queue for 2025-12-31 lists 8 items in this order (R6): UNDERPAD, SPECCUP, IVKIT, SALFLUSH, MASK, GAUZE, BLOODTUBE, GLOVEM.

## 9. Deliverables from the code generator
`app.py` (and `pages/` if used), `requirements.txt`, and `README.md`. Write clean, commented code. No placeholder or dummy data: everything comes from the CSVs above.
