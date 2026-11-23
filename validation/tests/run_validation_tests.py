#!/usr/bin/env python3
"""Automated validation tests for the ED Supply Decision Support app (v3.1).

Drives the real Streamlit app headlessly (streamlit.testing.v1.AppTest) and reconciles what it DISPLAYS with
independent pandas calculations from the CSV files (not with the app's own helper functions).

Run from the repository root:   python validation/tests/run_validation_tests.py
Outputs (evidence/outputs/): validation_test_results.csv, validation_test_results.json,
                             summary_text_samples.md, filter_reconciliation.csv
Matrix tests covered: TEST-15 (KPI/filter reconciliation), TEST-21 (Scenario Comparison report), TEST-22
(Recommended-actions reconciliation), TEST-06/TEST-14 (invalid input and generated summary), TEST-23 (boundary
what-if inputs), TEST-24 (summary language), TEST-25 (empty/unsupported filter states).
"""
from __future__ import annotations
import json, math, re, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apptest_helpers import open_page, close, metrics, ROOT

OUT = ROOT / "evidence" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
D = ROOT / "data"
arr = pd.read_csv(D / "raw/generated_ed_arrivals.csv", parse_dates=["date"])
daily = pd.read_csv(D / "final/inventory_decision_daily.csv", parse_dates=["date"])
scen = pd.read_csv(D / "final/scenario_comparison.csv", parse_dates=["date"])
items = pd.read_csv(D / "raw/item_assumptions.csv")
mdata = pd.read_csv(D / "modeling/model_dataset.csv", parse_dates=["date"])
evalp = pd.read_csv(D / "modeling/evaluation_predictions.csv", parse_dates=["date"])
meta = json.loads((ROOT / "models/model_metadata.json").read_text())
RESULTS: list[dict] = []
FILTER_ROWS: list[dict] = []
SAMPLES: list[str] = []


def rec(test, case, expected, observed, ok, note=""):
    RESULTS.append({"test_id": test, "case": case, "expected": str(expected), "observed": str(observed),
                    "status": "PASS" if ok else "FAIL", "note": note})
    print(("PASS " if ok else "FAIL ") + f"{test} | {case} | exp={expected} | obs={observed}")


def num(s: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", s))


# ---------------------------------------------------------------- independent KPI calculation
def indep_kpis(d0, d1, item_ids, thr):
    a = arr[(arr.date >= d0) & (arr.date <= d1)]
    f = daily[(daily.date >= d0) & (daily.date <= d1) & daily.item_id.isin(item_ids)]
    one = daily[(daily.date >= d0) & (daily.date <= d1)].drop_duplicates("date")
    out = {"Total ED arrivals": float(a.ed_arrivals.sum()), "Average daily arrivals": float(a.ed_arrivals.mean()),
           "Arrival forecast WAPE": float((one.ed_arrivals - one.forecast_arrivals_1d).abs().sum() / one.ed_arrivals.sum() * 100),
           "Surge days": float(a.surge_flag.sum())}
    if len(f):
        req = f.item_demand_requested.sum()
        out["Fill rate"] = float(np.minimum(f.item_issued_qty, f.item_demand_requested).sum() / req * 100)
        out["Backorder days"] = float((f.backorder_qty > 0).sum())
        out["High-risk item-days"] = float((f.stockout_risk_proxy >= thr).sum())
        out["Order spend"] = float((f.order_placed_qty * f.unit_cost).sum())
    return out, len(a), len(f)


TOL = {"Total ED arrivals": 0, "Average daily arrivals": 0.051, "Arrival forecast WAPE": 0.0051, "Surge days": 0,
       "Fill rate": 0.0051, "Backorder days": 0, "High-risk item-days": 0, "Order spend": 0.51}


def apply_filters(at, d0, d1, item_ids, thr):
    at.sidebar.date_input[0].set_value((d0, d1))
    at.sidebar.slider[0].set_value(thr)
    at.sidebar.multiselect[1].set_value(item_ids)
    at.run()
    return at


def sidebar_caption(at):
    for c in at.sidebar.caption:
        m = re.match(r"([\d,]+) days · (\d+) of (\d+) items · ([\d,]+) item-day rows", c.value)
        if m:
            return [int(x.replace(",", "")) for x in m.groups()]
    return None


ALL = list(items.item_id)
CASES = [
    ("N1", "normal: full range, all items, threshold 0.50", dt.date(2024, 1, 1), dt.date(2025, 12, 31), ALL, 0.50),
    ("N2", "normal: Mar 2025, GAUZE+MASK, threshold 0.30", dt.date(2025, 3, 1), dt.date(2025, 3, 31), ["GAUZE", "MASK"], 0.30),
    ("B1", "boundary: single first day 2024-01-01", dt.date(2024, 1, 1), dt.date(2024, 1, 1), ALL, 0.50),
    ("B2", "boundary: single last day 2025-12-31", dt.date(2025, 12, 31), dt.date(2025, 12, 31), ALL, 0.50),
    ("B3", "boundary: threshold 0.00 (every item-day high-risk)", dt.date(2025, 6, 1), dt.date(2025, 6, 30), ["IVKIT"], 0.00),
    ("B4", "boundary: threshold 1.00", dt.date(2025, 6, 1), dt.date(2025, 6, 30), ["IVKIT"], 1.00),
]

# ================================================================= TEST-15: KPI cards + cross-page filters
for cid, label, d0, d1, ids, thr in CASES:
    at = open_page("overview"); apply_filters(at, d0, d1, ids, thr)
    exp, n_days, n_rows = indep_kpis(pd.Timestamp(d0), pd.Timestamp(d1), ids, thr)
    shown = {k: num(v) for k, v in metrics(at).items()}
    for k, e in exp.items():
        o = shown.get(k)
        ok = o is not None and abs(o - e) <= TOL[k]
        rec("TEST-15", f"{cid} Overview KPI '{k}' [{label}]", f"{e:.4f}", o, ok)
        FILTER_ROWS.append({"case": cid, "page": "Overview", "item": k, "independent": round(e, 4), "displayed": o, "match": ok})
    rec("TEST-15", f"{cid} no exceptions on Overview", 0, len(at.exception), len(at.exception) == 0)
    cap = sidebar_caption(at)
    rec("TEST-15", f"{cid} active-population caption [{label}]", f"{n_days} days, {len(ids)} items, {n_rows} rows", cap,
        cap is not None and cap[0] == n_days and cap[1] == len(ids) and cap[3] == n_rows)
    close(at)
    if cid in ("N1", "N2", "B1"):
        for page in ("item_explorer", "trends", "scenarios", "system_flow", "data", "predictive"):
            at = open_page(page); apply_filters(at, d0, d1, ids, thr)
            cap = sidebar_caption(at)
            ok = cap is not None and cap[0] == n_days and cap[3] == n_rows
            rec("TEST-15", f"{cid} same filter population on {page}", f"{n_days} days / {n_rows} rows", cap, ok)
            FILTER_ROWS.append({"case": cid, "page": page, "item": "active population caption", "independent": f"{n_days}/{n_rows}", "displayed": cap, "match": ok})
            if page == "data":
                dcap = [c.value for c in at.caption if "rows match the current filters" in c.value]
                rec("TEST-15", f"{cid} Data page download row count", f"{n_rows:,} rows match", dcap[0] if dcap else None,
                    bool(dcap) and num(dcap[0].split(" rows")[0]) == n_rows)
            if page == "item_explorer" and len(ids):
                it = at.selectbox[0].value
                f = daily[(daily.date >= pd.Timestamp(d0)) & (daily.date <= pd.Timestamp(d1)) & (daily.item_id == it)]
                m = {k: v for k, v in metrics(at).items()}
                checks = {"Avg daily demand": f.item_demand_requested.mean(), "High-risk days": (f.stockout_risk_proxy >= thr).sum(),
                          "Supplier disruptions": (f.supplier_event != "none").sum(), "Order spend": (f.order_placed_qty * f.unit_cost).sum()}
                for k, e in checks.items():
                    o = num(m[k]); tol = 0.51 if k == "Order spend" else (0.051 if k == "Avg daily demand" else 0)
                    ok = abs(o - e) <= tol
                    rec("TEST-15", f"{cid} Item Explorer '{k}' ({it})", f"{e:.2f}", o, ok)
                    FILTER_ROWS.append({"case": cid, "page": "Item Explorer", "item": f"{k} ({it})", "independent": round(float(e), 2), "displayed": o, "match": ok})
            close(at)

# ================================================================= TEST-25: unsupported / empty filter states
at = open_page("overview"); at.sidebar.multiselect[1].set_value([]); at.run()
m = metrics(at)
item_kpis = ["Fill rate", "Backorder days", "High-risk item-days", "Order spend"]
rec("TEST-25", "Overview with no items selected: item KPIs show an em dash instead of a number", "— for 4 item KPIs",
    {k: m.get(k) for k in item_kpis}, all(m.get(k) == "—" for k in item_kpis) and len(at.exception) == 0)
rec("TEST-25", "Overview with no items selected: arrival KPIs still correct", "141,638 arrivals", m.get("Total ED arrivals"), m.get("Total ED arrivals") == "141,638")
close(at)
at = open_page("overview"); at.sidebar.date_input[0].set_value((dt.date(2025, 3, 1),)); at.run()
warn = [w.value for w in at.sidebar.warning]
rec("TEST-25", "Incomplete date range: warning shown and dashboard withheld", "warning + no KPI cards",
    {"warnings": warn, "kpi_cards": len(at.metric)}, bool(warn) and len(at.metric) == 0 and len(at.exception) == 0)
close(at)
at = open_page("scenarios"); at.sidebar.multiselect[1].set_value([]); at.run()
msgs = [i.value for i in at.info]
rec("TEST-25", "Scenario Comparison with no items: explanatory message, no table", "message", msgs[:1], bool(msgs) and len(at.dataframe) == 0)
close(at)
at = open_page("predictive"); at.sidebar.date_input[0].set_value((dt.date(2025, 12, 25), dt.date(2025, 12, 31))); at.run()
msgs = [i.value for i in at.info]
rec("TEST-25", "Predictive page for dates after 2025-12-24 (no model rows): explains why nothing is shown", "no-model-rows message",
    [x[:90] for x in msgs if "model rows" in x.lower()][:1], any("model rows" in x.lower() for x in msgs) and len(at.exception) == 0)
close(at)

# ================================================================= TEST-21: Scenario Comparison report
unit_cost = items.set_index("item_id").unit_cost
for cid, d, it in [("N", dt.date(2025, 7, 15), "GAUZE"), ("B-first", dt.date(2024, 1, 1), "IVKIT"), ("B-last", dt.date(2025, 12, 31), "GLOVEM")]:
    at = open_page("scenarios")
    at.sidebar.date_input[0].set_value((dt.date(2024, 1, 1), dt.date(2025, 12, 31)))
    at.run()
    at.date_input[0].set_value(d); at.selectbox[0].set_value(it); at.run()
    t = at.dataframe[0].value
    s = scen[(scen.date == pd.Timestamp(d)) & (scen.item_id == it)].set_index("scenario").loc[["low", "base", "high"]]
    ok_vals = (list(t["Scenario"]) == ["low", "base", "high"]
               and np.allclose(t["7-day item demand"].astype(float), s.forecast_item_demand_7d, atol=0.051)
               and np.allclose(t["Recommended order qty"].astype(float), s.recommended_order_qty)
               and np.allclose(t["Estimated cost"].astype(float), s.recommended_order_qty * unit_cost[it], atol=0.006)
               and np.allclose(t["Demand multiplier"].astype(float), s.demand_multiplier))
    rec("TEST-21", f"{cid}: displayed low/base/high rows equal scenario_comparison.csv ({it}, {d})", "exact (display rounding)", "match" if ok_vals else t.to_dict("list"), ok_vals)
    mono = bool(t["7-day item demand"].is_monotonic_increasing and t["Days of supply"].is_monotonic_decreasing and t["Risk score"].is_monotonic_increasing)
    rec("TEST-21", f"{cid}: demand rises and days-of-supply falls from low to high", "monotonic", mono, mono)
    rec("TEST-21", f"{cid}: physical stock identical across scenarios (caption claim)", "1 distinct inventory position", t["Inventory position"].nunique(), t["Inventory position"].nunique() == 1)
    close(at)

# ================================================================= TEST-22: Recommended actions vs independent calc
ev = evalp.set_index(["date", "item_id"])
thr_s = meta["shortage"]["decision_threshold"]
for cid, d in [("N", dt.date(2025, 6, 10)), ("B-first model date", dt.date(2025, 1, 1)), ("B-last model date", dt.date(2025, 12, 24))]:
    at = open_page("predictive")
    at.sidebar.date_input[0].set_value((dt.date(2025, 1, 1), dt.date(2025, 12, 31))); at.run()
    at.selectbox(key="rec_date").set_value(d); at.run()
    shown = [x.value for x in at.dataframe if "Suggested order (units)" in x.value.columns][0]
    rows = mdata[mdata.date == pd.Timestamp(d)]
    exp = []
    for _, r in rows.iterrows():
        dd = daily[(daily.date == r.date) & (daily.item_id == r.item_id)].iloc[0]
        demand7 = r.forecast_arrivals_7d * r.use_rate_per_arrival
        gap = max(0.0, demand7 - dd.inventory_position)
        units = int(math.ceil(gap / dd.case_pack) * dd.case_pack) if gap > 0 else 0
        p = float(ev.loc[(r.date, r.item_id), "shortage_probability"])
        cat = "Review now" if p >= thr_s else ("Order this week" if units > 0 else "No action needed")
        exp.append((items.set_index("item_id").item_name[r.item_id], cat, units, p))
    got = {row["Item"]: (row["Recommended action"], int(row["Suggested order (units)"].split(" ")[0].replace(",", "")), row["Shortage probability"]) for _, row in shown.iterrows()}
    bad = []
    for name, cat, units, p in exp:
        g = got.get(name)
        if not g or cat not in g[0] or g[1] != units or abs(num(g[2]) - p * 100) > 0.051:
            bad.append((name, cat, units, round(p * 100, 1), g))
    rec("TEST-22", f"{cid} ({d}): all {len(exp)} items: category, whole-case order quantity and probability match independent calc", "0 mismatches", f"{len(bad)} mismatches {bad[:2]}", not bad)
    cp = {int(r["Suggested order (units)"].split(" ")[0].replace(",", "")) % int(daily[(daily.date == pd.Timestamp(d)) & (daily.item_id == mdata.item_id.iloc[0])].case_pack.iloc[0]) for _, r in shown.iterrows()}
    close(at)

# ================================================================= What-if: normal, boundary, invalid + summary text
def whatif_run(model: str, item: str, d: dt.date, edits: dict, label_key: str, start: dt.date = dt.date(2025, 1, 1)):
    at = open_page("predictive")
    at.sidebar.date_input[0].set_value((start, dt.date(2025, 12, 31))); at.run()
    if model == "shortage":
        at.radio[0].set_value("Seven-day shortage probability"); at.run()
    key = model
    at.selectbox(key=f"model_item_{key}").set_value(item); at.run()
    at.selectbox(key=f"model_date_{key}").set_value(d); at.run()
    ni = {n.label: n for n in at.number_input}
    for lbl, val in edits.items():
        [n for n in at.number_input if n.label.startswith(lbl)][0].set_value(val)
    [b for b in at.button if "Run model-based scenario" in b.label][0].click(); at.run()
    return at


def parse_cmp(at):
    df = [x.value for x in at.dataframe if "Baseline" in x.value.columns and "Scenario" in x.value.columns]
    return df[-1] if df else None


# --- demand: normal (IVKIT 2025-12-24 : 1285.3 -> 1485.3 / 1210.3) --------------------------------
base_row = mdata[(mdata.item_id == "IVKIT") & (mdata.date == "2025-12-24")].iloc[0]
use = base_row.use_rate_per_arrival; F0 = base_row.forecast_arrivals_7d
rng = meta["demand"]["input_ranges"]["forecast_arrivals_7d"]
for cid, F1 in [("normal-high", 1485.3), ("normal-low", 1210.3), ("boundary-max", rng["max"]), ("boundary-min", rng["min"])]:
    at = whatif_run("demand", "IVKIT", dt.date(2025, 12, 24), {"Seven-day arrival forecast": F1}, "")
    cmpdf = parse_cmp(at)
    summ = [s.value for s in at.success]
    base_u, new_u = F0 * use, F1 * use
    if cmpdf is not None:
        pred = cmpdf[cmpdf[""].str.contains("Predicted")].iloc[0]
        ok = num(pred["Baseline"]) == round(base_u) and num(pred["Scenario"]) == round(new_u) and abs(num(pred["Difference"]) - (round(new_u) - round(base_u))) <= 1
        obs = f"{pred['Baseline']} -> {pred['Scenario']} ({pred['Difference']})"
    else:
        ok, obs = False, f"no comparison table; errors={[e.value[:80] for e in at.error]}"
    rec("TEST-02" if cid.startswith("normal") else "TEST-23", f"demand what-if {cid}: 7-day forecast {F0:.1f} -> {F1:.1f}; independent {base_u:.3f} -> {new_u:.3f} units", f"{round(base_u)} -> {round(new_u)}", obs, ok)
    if summ:
        text = summ[0]
        m = re.search(r"moved from ([\d,]+) to ([\d,]+) units \(([+\-][\d,]+) units, ([+\-][\d.]+)%\)", text)
        exp_pct = (new_u - base_u) / base_u * 100 if base_u else 0
        # Summary uses the rounded values displayed to the user
        ok2 = bool(m) and num(m.group(1)) == round(base_u) and num(m.group(2)) == round(new_u)
        rec("TEST-14", f"demand summary figures match independent calculation ({cid})", f"{round(base_u):,} -> {round(new_u):,}", m.group(0) if m else text[:120], ok2)
        causal = re.findall(r"\b(causes?|caused|leads? to|will result|proves?|due to)\b", text, flags=re.I)
        has_hedge = ("model-based" in text and "not a proven or guaranteed" in text)
        rec("TEST-24", f"demand summary ({cid}): limitation language present, no causal wording", "hedge present; 0 causal terms", f"hedge={has_hedge}; causal terms={causal}", has_hedge and not causal)
        SAMPLES.append(f"## Demand what-if — {cid} (IVKIT, 2025-12-24, 7-day arrival forecast {F0:.1f} → {F1:.1f})\n\n{text}\n")
    close(at)

# --- demand: invalid input (ratio check) -------------------------------------------------------
dd = mdata.copy()
cand_hi = dd[(dd.item_id == "IVKIT") & (rng["max"] / (7 * dd.forecast_arrivals_1d) > 1.40)]
cand_lo = dd[(dd.item_id == "IVKIT") & (dd.date >= "2025-01-01") & (rng["min"] / (7 * dd.forecast_arrivals_1d) < 0.75)]
for cid, cand, F1 in [("invalid-high-ratio", cand_hi, rng["max"]), ("invalid-low-ratio", cand_lo, rng["min"])]:
    if cand.empty:
        rec("TEST-06", f"demand {cid}: found a date where the invalid band is reachable", "date found", "none in 2025 IVKIT", False); continue
    r = cand.iloc[0]; ratio = F1 / (7 * r.forecast_arrivals_1d)
    at = whatif_run("demand", "IVKIT", r.date.date(), {"Seven-day arrival forecast": F1}, "", start=dt.date(2024, 1, 1))
    errs = [e.value for e in at.error]; has_table = parse_cmp(at) is not None; no_summary = len(at.success) == 0
    infos = [i.value for i in at.info if "No scenario prediction was produced" in i.value]
    ok = bool(errs) and not has_table and no_summary and bool(infos) and f"{ratio:.2f}×" in errs[0]
    rec("TEST-06", f"demand {cid} ({r.date.date()}, 7d={F1:.1f}, 1d forecast={r.forecast_arrivals_1d:.1f}, ratio={ratio:.2f}): error shown, NO prediction, NO summary", "error + 'No scenario prediction was produced'",
        f"errors={len(errs)}, comparison_table={has_table}, summary={not no_summary}; msg={errs[0][:140] if errs else None}", ok)
    SAMPLES.append(f"## Demand what-if — {cid} (IVKIT, {r.date.date()}, 7-day arrival forecast {F1:.1f}; one-day forecast {r.forecast_arrivals_1d:.1f})\n\n**Error shown:** {errs[0] if errs else '(none)'}\n\n**Info shown:** {infos[0] if infos else '(none)'}\n\n(No prediction table or summary was displayed.)\n")
    close(at)

# --- ratio rule: inclusive boundaries (unit level, same function the app calls) ----------------------
sys.path.insert(0, str(ROOT / "code"))
import model_features as MF
lo_lim, hi_lim = meta["demand"]["forecast_ratio_limits"]
for label, f7, f1, expect_ok in [("exactly 0.75x", 0.75 * 7 * 200, 200, True), ("exactly 1.40x", 1.40 * 7 * 200, 200, True),
                                 ("just below 0.75x", 0.7499 * 7 * 200, 200, False), ("just above 1.40x", 1.4001 * 7 * 200, 200, False)]:
    r_ = MF.forecast_ratio(f7, f1); accepted = lo_lim <= r_ <= hi_lim
    rec("TEST-06", f"ratio rule boundary: {label} (limits {lo_lim}-{hi_lim})", "accepted" if expect_ok else "rejected", "accepted" if accepted else "rejected", accepted == expect_ok)

# --- shortage: normal and boundary ---------------------------------------------------------------
srow = mdata[(mdata.item_id == "GAUZE") & (mdata.date == "2025-12-24")].iloc[0]
sr = meta["shortage"]["input_ranges_by_item"]["GAUZE"]["closing_on_hand"]
probs = {}
for cid, stock in [("normal-lower-stock", max(sr["min"], srow.closing_on_hand * 0.5)), ("boundary-min-stock", sr["min"]), ("boundary-max-stock", sr["max"])]:
    at = whatif_run("shortage", "GAUZE", dt.date(2025, 12, 24), {"Closing stock": int(stock)}, "")
    cmpdf = parse_cmp(at); summ = [s.value for s in at.success]
    if cmpdf is not None:
        pred = cmpdf[cmpdf[""].str.contains("Predicted shortage")].iloc[0]
        b, s_ = num(pred["Baseline"]), num(pred["Scenario"]); probs[cid] = s_
        base_ref = float(ev.loc[(pd.Timestamp("2025-12-24"), "GAUZE"), "shortage_probability"]) * 100
        ok = abs(b - base_ref) <= 0.051 and (s_ >= b - 0.051 if stock <= srow.closing_on_hand else s_ <= b + 0.051)
        rec("TEST-23", f"shortage what-if {cid}: closing stock {srow.closing_on_hand:.0f} -> {int(stock)}; baseline equals saved evaluation prediction and direction is right",
            f"baseline {base_ref:.1f}%; lower stock => not lower risk", f"{pred['Baseline']} -> {pred['Scenario']} ({pred['Difference']})", ok)
        if summ:
            text = summ[0]
            m = re.search(r"moved from ([\d.]+)% to ([\d.]+)% \(([+\-][\d.]+) percentage points\)", text)
            ok2 = bool(m) and abs(float(m.group(1)) - b) <= 0.051 and abs(float(m.group(2)) - s_) <= 0.051
            rec("TEST-14", f"shortage summary figures equal side-by-side table ({cid})", f"{b}% -> {s_}%", m.group(0) if m else text[:100], ok2)
            causal = re.findall(r"\b(causes?|caused|leads? to|will result|proves?|due to)\b", text, flags=re.I)
            hedge = "model-based scenario" in text and "not a proven or guaranteed" in text
            rec("TEST-24", f"shortage summary ({cid}): limitation + trade-off language, no causal wording", "hedge + trade-offs; 0 causal terms", f"hedge={hedge}; trade-offs={'Trade-offs' in text}; causal={causal}", hedge and "Trade-offs" in text and not causal)
            SAMPLES.append(f"## Shortage what-if — {cid} (GAUZE, 2025-12-24, closing stock {srow.closing_on_hand:.0f} → {int(stock)})\n\n{text}\n")
    else:
        rec("TEST-23", f"shortage what-if {cid}", "comparison table", f"none; errors={[e.value[:100] for e in at.error]}", False)
    close(at)

# ================================================================= TEST-26: limitation / uncertainty messages shown to the user
need = {"Seven-day item demand": ["surge", "11.1%", "safety stock", "synthetic", "optimistic", "Not for real ordering"],
        "Seven-day shortage probability": ["rare", "false alarm", "not as an automatic ordering", "only 3 items", "unstable", "synthetic", "Not for real ordering"]}
for mod, phrases in need.items():
    at = open_page("predictive")
    if "shortage" in mod:
        at.radio[0].set_value(mod); at.run()
    blob = " ".join([w.value for w in at.warning] + [i.value for i in at.info]).lower()
    missing = [ph for ph in phrases if ph.lower() not in blob]
    rec("TEST-26", f"{mod}: page displays required limitation and uncertainty statements", f"all {len(phrases)} phrases present", f"missing={missing}", not missing)
    close(at)

# ---------------------------------------------------------------------------- write outputs
df = pd.DataFrame(RESULTS)
df.to_csv(OUT / "validation_test_results.csv", index=False)
(OUT / "validation_test_results.json").write_text(json.dumps(RESULTS, indent=1))
pd.DataFrame(FILTER_ROWS).to_csv(OUT / "filter_reconciliation.csv", index=False)
(OUT / "summary_text_samples.md").write_text("# Saved samples of the automatically generated scenario summaries and invalid-input messages\n\nCaptured by `validation/tests/run_validation_tests.py` from the running app.\n\n" + "\n".join(SAMPLES))
print("\nSUMMARY by test:\n", df.groupby("test_id").status.value_counts().unstack(fill_value=0))
print("TOTAL", len(df), "FAIL", int((df.status == "FAIL").sum()))
