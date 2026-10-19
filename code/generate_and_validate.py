#!/usr/bin/env python3
"""Generate, transform, and validate the prototype dataset (v1.1 demo data).

Changes from the Week 7 submission (v1.0):
  1. Arrival surge episodes (multi-day respiratory-season surges and one-day spikes).
  2. Periodic-review ordering: orders are placed only on each item's review days
     (twice weekly); the recommendation is still computed every day.
  3. Supplier disruptions: some orders arrive late or short-shipped.
  4. Starting stock equals target stock (removes the artificial day-1 escalations).
  5. New columns: order_placed_qty, supplier_event (daily table); surge_flag (arrivals).

Original description:

The pipeline intentionally uses synthetic data. It creates a transparent,
reproducible item-day dataset for a single emergency department; it does not
represent a real hospital or real patients.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


SEED = 20261009
START = date(2024, 1, 1)
END = date(2025, 12, 31)
ASSUMPTION_VERSION = "v1.1-2026-10-19"

# v1.1 simulation assumptions (illustrative, stated explicitly).
REVIEW_WEEKDAYS = (0, 3)          # orders placed Monday and Thursday
REVIEW_PERIOD_DAYS = 4            # longest gap between review days
P_DELAY = 0.08                    # chance an order arrives late
DELAY_DAYS = (2, 6)               # extra days when late
P_SHORT = 0.05                    # chance an order is short-shipped
SHORT_FILL = (0.3, 0.7)           # share of ordered units actually shipped
N_SURGES = 6                      # multi-day surge episodes over two years
SURGE_LEN = (4, 10)
SURGE_LIFT = (1.18, 1.38)
N_SPIKES = 8                      # one-day spikes
SPIKE_LIFT = (1.45, 1.70)

ITEMS = [
    {"item_id": "IVKIT", "item_name": "IV start kit", "category": "vascular access", "use_rate": 0.42, "use_cv": 0.20, "lead_time_days": 3, "case_pack": 24, "unit_cost": 3.80, "service_level_target": 0.97, "safety_stock_days": 3},
    {"item_id": "GAUZE", "item_name": "Sterile gauze pack", "category": "wound care", "use_rate": 1.15, "use_cv": 0.18, "lead_time_days": 2, "case_pack": 100, "unit_cost": 0.34, "service_level_target": 0.96, "safety_stock_days": 2},
    {"item_id": "GLOVEM", "item_name": "Exam gloves, medium (pair)", "category": "PPE", "use_rate": 3.60, "use_cv": 0.15, "lead_time_days": 3, "case_pack": 200, "unit_cost": 0.18, "service_level_target": 0.98, "safety_stock_days": 4},
    {"item_id": "SPECCUP", "item_name": "Specimen cup", "category": "laboratory", "use_rate": 0.36, "use_cv": 0.24, "lead_time_days": 4, "case_pack": 50, "unit_cost": 0.62, "service_level_target": 0.95, "safety_stock_days": 3},
    {"item_id": "SALFLUSH", "item_name": "Saline flush 10 mL", "category": "vascular access", "use_rate": 0.88, "use_cv": 0.20, "lead_time_days": 5, "case_pack": 100, "unit_cost": 0.47, "service_level_target": 0.98, "safety_stock_days": 5},
    {"item_id": "MASK", "item_name": "Procedure mask", "category": "PPE", "use_rate": 0.74, "use_cv": 0.22, "lead_time_days": 3, "case_pack": 50, "unit_cost": 0.21, "service_level_target": 0.96, "safety_stock_days": 3},
    {"item_id": "UNDERPAD", "item_name": "Disposable underpad", "category": "patient care", "use_rate": 0.52, "use_cv": 0.25, "lead_time_days": 4, "case_pack": 50, "unit_cost": 0.71, "service_level_target": 0.95, "safety_stock_days": 3},
    {"item_id": "BLOODTUBE", "item_name": "Blood collection tube", "category": "laboratory", "use_rate": 1.32, "use_cv": 0.18, "lead_time_days": 3, "case_pack": 100, "unit_cost": 0.29, "service_level_target": 0.97, "safety_stock_days": 3},
]


def dates_between(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def poisson(lam: float, rng: random.Random) -> int:
    """Dependency-free Poisson draw; normal approximation for large lambda."""
    if lam < 30:
        limit, product, k = math.exp(-lam), 1.0, 0
        while product > limit:
            k += 1
            product *= rng.random()
        return max(0, k - 1)
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def ceil_pack(qty: float, pack: int) -> int:
    return 0 if qty <= 0 else int(math.ceil(qty / pack) * pack)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arrival_mean(d: date) -> float:
    # Baseline and exact multipliers are explicit prototype assumptions.
    dow = [1.04, 1.03, 1.01, 1.00, 1.02, 0.94, 0.91][d.weekday()]
    annual = 1 + 0.07 * math.sin(2 * math.pi * (d.timetuple().tm_yday - 20) / 365.25)
    holiday = 0.88 if (d.month, d.day) in {(1, 1), (7, 4), (12, 25)} else 1.0
    trend = 1 + 0.00008 * (d - START).days
    return 185.0 * dow * annual * holiday * trend


def forecast_arrivals(target: date, history: dict[date, int]) -> float:
    same_dow = [v for d, v in history.items() if d < target and d.weekday() == target.weekday() and (target - d).days <= 56]
    recent = [v for d, v in history.items() if d < target and (target - d).days <= 28]
    values = same_dow[-8:] or recent[-28:]
    return round(sum(values) / len(values), 1) if values else round(arrival_mean(target), 1)


def risk_proxy(inventory_position: float, demand_mean: float, demand_sd: float) -> float:
    if demand_sd <= 0:
        return float(inventory_position < demand_mean)
    z = (inventory_position - demand_mean) / demand_sd
    return round(1.0 / (1.0 + math.exp(1.7 * z)), 4)


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    intermediate_dir = root / "data" / "intermediate"
    final_dir = root / "data" / "final"
    docs_dir = root / "docs"
    for p in (raw_dir, intermediate_dir, final_dir, docs_dir):
        p.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    days = list(dates_between(START, END))

    surge_rng = random.Random(SEED + 1)
    surge = {}
    for _ in range(N_SURGES):
        start_i = surge_rng.randrange(14, len(days) - 12)
        lift = surge_rng.uniform(*SURGE_LIFT)
        for k in range(surge_rng.randint(*SURGE_LEN)):
            surge[days[start_i + k]] = max(surge.get(days[start_i + k], 1.0), lift)
    for _ in range(N_SPIKES):
        dd = days[surge_rng.randrange(14, len(days))]
        surge[dd] = max(surge.get(dd, 1.0), surge_rng.uniform(*SPIKE_LIFT))

    arrivals = []
    arrival_history: dict[date, int] = {}
    for d in days:
        expected = arrival_mean(d) * surge.get(d, 1.0)
        count = poisson(expected, rng)
        arrival_history[d] = count
        arrivals.append({
            "date": d.isoformat(), "ed_arrivals": count,
            "expected_arrivals_parameter": round(expected, 2),
            "day_of_week": d.strftime("%A"), "weekend_flag": int(d.weekday() >= 5),
            "holiday_flag": int((d.month, d.day) in {(1, 1), (7, 4), (12, 25)}),
            "surge_flag": int(d in surge),
            "source_status": "generated"
        })
    write_csv(raw_dir / "generated_ed_arrivals.csv", arrivals)

    item_rows = []
    for item in ITEMS:
        r = dict(item)
        r.update({"assumption_version": ASSUMPTION_VERSION, "source_status": "manually_constructed_assumption"})
        item_rows.append(r)
    write_csv(raw_dir / "item_assumptions.csv", item_rows)

    final_rows = []
    scenario_rows = []
    supply_rng = random.Random(SEED + 2)
    orders: list[dict] = []   # each: item, qty_ordered, qty_shipped, expected, actual, received
    on_hand = {}
    backorders = defaultdict(int)
    for item in ITEMS:
        daily = 185 * item["use_rate"]
        # v1.1: start at target stock; exceptions now come from surges and supplier events.
        on_hand[item["item_id"]] = ceil_pack(daily * (item["lead_time_days"] + item["safety_stock_days"] + REVIEW_PERIOD_DAYS), item["case_pack"])

    for d in days:
        next7 = [d + timedelta(days=i) for i in range(1, 8)]
        forecast_1d = forecast_arrivals(d + timedelta(days=1), {k: v for k, v in arrival_history.items() if k <= d})
        forecast_7d = round(sum(forecast_arrivals(x, {k: v for k, v in arrival_history.items() if k <= d}) for x in next7), 1)
        for item in ITEMS:
            iid = item["item_id"]
            opening = on_hand[iid]
            todays = [o for o in orders if o["item"] == iid and o["actual"] == d]
            receipts = sum(o["qty_shipped"] for o in todays)
            events = []
            if any(o["qty_shipped"] < o["qty_ordered"] for o in todays):
                events.append("short_shipped")
            if any(o["expected"] == d and o["actual"] > d for o in orders if o["item"] == iid):
                events.append("delayed")
            supplier_event = "+".join(events) or "none"
            available = opening + receipts
            lam = arrival_history[d] * item["use_rate"]
            requested = poisson(lam, rng)
            total_need = requested + backorders[iid]
            issued = min(available, total_need)
            backorder = total_need - issued
            closing = available - issued
            backorders[iid] = backorder

            # Planner's view: late orders stay open; short shipments are only known on arrival.
            future_open = sum(o["qty_ordered"] for o in orders if o["item"] == iid and o["actual"] > d)
            inventory_position = closing + future_open - backorder
            forecast_demand_7d = round(forecast_7d * item["use_rate"], 1)
            avg_daily = max(0.1, forecast_demand_7d / 7)
            days_supply = round(max(0, inventory_position) / avg_daily, 2)
            protection_days = item["lead_time_days"] + item["safety_stock_days"]
            protection_demand = avg_daily * protection_days
            demand_sd = max(1.0, math.sqrt(protection_demand) * (1 + item["use_cv"]))
            risk = risk_proxy(inventory_position, protection_demand, demand_sd)
            target_stock = avg_daily * (protection_days + REVIEW_PERIOD_DAYS)
            recommended = ceil_pack(target_stock - inventory_position, item["case_pack"])
            action = "escalate" if risk >= 0.50 else ("order" if recommended > 0 else "monitor")

            placed = recommended if (d.weekday() in REVIEW_WEEKDAYS or risk >= 0.50) else 0
            if placed > 0:
                expected = d + timedelta(days=item["lead_time_days"])
                actual = expected
                shipped = placed
                if supply_rng.random() < P_DELAY:
                    actual = expected + timedelta(days=supply_rng.randint(*DELAY_DAYS))
                elif supply_rng.random() < P_SHORT:
                    shipped = int(round(placed * supply_rng.uniform(*SHORT_FILL)))
                orders.append({"item": iid, "qty_ordered": placed, "qty_shipped": shipped,
                               "expected": expected, "actual": actual})

            final_rows.append({
                "date": d.isoformat(), "item_id": iid, "ed_arrivals": arrival_history[d],
                "forecast_arrivals_1d": forecast_1d, "forecast_arrivals_7d": forecast_7d,
                "item_demand_requested": requested, "item_issued_qty": issued,
                "backorder_qty": backorder, "opening_on_hand": opening, "receipts_qty": receipts,
                "closing_on_hand": closing, "open_order_qty": future_open,
                "inventory_position": inventory_position, "forecast_item_demand_7d": forecast_demand_7d,
                "days_of_supply": days_supply, "stockout_risk_proxy": risk,
                "recommended_order_qty": recommended, "recommended_action": action,
                "order_placed_qty": placed, "supplier_event": supplier_event,
                "lead_time_days": item["lead_time_days"], "case_pack": item["case_pack"],
                "unit_cost": item["unit_cost"], "estimated_order_cost": round(recommended * item["unit_cost"], 2),
                "use_rate_per_arrival": item["use_rate"], "service_level_target": item["service_level_target"],
                "assumption_version": ASSUMPTION_VERSION,
                "record_status": "generated_and_derived"
            })
            on_hand[iid] = closing

            for scenario, multiplier in (("low", 0.85), ("base", 1.0), ("high", 1.20)):
                scen_demand = forecast_demand_7d * multiplier
                scen_avg = max(0.1, scen_demand / 7)
                scen_protection = scen_avg * protection_days
                scen_sd = max(1.0, math.sqrt(scen_protection) * (1 + item["use_cv"]))
                scen_risk = risk_proxy(inventory_position, scen_protection, scen_sd)
                scen_order = ceil_pack(scen_avg * (protection_days + REVIEW_PERIOD_DAYS) - inventory_position, item["case_pack"])
                scenario_rows.append({
                    "date": d.isoformat(), "item_id": iid, "scenario": scenario,
                    "demand_multiplier": multiplier, "forecast_item_demand_7d": round(scen_demand, 1),
                    "inventory_position": inventory_position,
                    "days_of_supply": round(max(0, inventory_position) / scen_avg, 2),
                    "stockout_risk_proxy": scen_risk, "recommended_order_qty": scen_order,
                    "assumption_version": ASSUMPTION_VERSION
                })

    write_csv(intermediate_dir / "item_day_operations.csv", final_rows)
    write_csv(final_dir / "inventory_decision_daily.csv", final_rows)
    write_csv(final_dir / "scenario_comparison.csv", scenario_rows)

    # Validation outcomes are computed, not hard-coded.
    checks = []
    def add(check, passed, observed, expected, action="None required"):
        checks.append({"check_id": f"V{len(checks)+1:02d}", "validation_check": check,
                       "status": "PASS" if passed else "FAIL", "observed": str(observed),
                       "expected": str(expected), "action_taken": action})

    required = {"date", "item_id", "ed_arrivals", "closing_on_hand", "stockout_risk_proxy", "recommended_order_qty"}
    add("Required schema", required.issubset(final_rows[0]), len(required.intersection(final_rows[0])), len(required))
    keys = [(r["date"], r["item_id"]) for r in final_rows]
    add("Item-day key uniqueness", len(keys) == len(set(keys)), len(set(keys)), len(keys))
    valid_items = {i["item_id"] for i in ITEMS}
    add("Item foreign-key integrity", all(r["item_id"] in valid_items for r in final_rows), sum(r["item_id"] not in valid_items for r in final_rows), 0)
    add("Missing required values", all(r[k] != "" and r[k] is not None for r in final_rows for k in required), 0, 0)
    add("Date coverage", final_rows[0]["date"] == START.isoformat() and final_rows[-1]["date"] == END.isoformat(), f"{final_rows[0]['date']} to {final_rows[-1]['date']}", f"{START} to {END}")
    add("Nonnegative physical quantities", all(r[k] >= 0 for r in final_rows for k in ("opening_on_hand", "receipts_qty", "item_demand_requested", "item_issued_qty", "backorder_qty", "closing_on_hand", "recommended_order_qty")), 0, 0)
    reconciliation_errors = sum(r["closing_on_hand"] != r["opening_on_hand"] + r["receipts_qty"] - r["item_issued_qty"] for r in final_rows)
    add("Daily inventory reconciliation", reconciliation_errors == 0, reconciliation_errors, 0)
    add("Issued quantity logical bound", all(r["item_issued_qty"] <= r["opening_on_hand"] + r["receipts_qty"] for r in final_rows), 0, 0)
    add("Risk range", all(0 <= r["stockout_risk_proxy"] <= 1 for r in final_rows), 0, 0)
    pack_map = {i["item_id"]: i["case_pack"] for i in ITEMS}
    pack_errors = sum(r["recommended_order_qty"] % pack_map[r["item_id"]] != 0 for r in final_rows)
    add("Case-pack rounding", pack_errors == 0, pack_errors, 0)
    scen_key = {(r["date"], r["item_id"], r["scenario"]) for r in scenario_rows}
    add("Scenario key uniqueness", len(scen_key) == len(scenario_rows), len(scen_key), len(scenario_rows))
    by_key = defaultdict(dict)
    for r in scenario_rows:
        by_key[(r["date"], r["item_id"])][r["scenario"]] = r
    mono_errors = sum(not (v["low"]["stockout_risk_proxy"] <= v["base"]["stockout_risk_proxy"] <= v["high"]["stockout_risk_proxy"]) for v in by_key.values())
    add("Scenario risk monotonicity", mono_errors == 0, mono_errors, 0)
    metric_fields = {"forecast_arrivals_7d", "forecast_item_demand_7d", "inventory_position", "days_of_supply", "stockout_risk_proxy", "recommended_order_qty"}
    add("Planned metric computability", metric_fields.issubset(final_rows[0]), len(metric_fields.intersection(final_rows[0])), len(metric_fields))
    write_csv(final_dir / "validation_results.csv", checks)

    dictionary = [
        ("date", "inventory_decision_daily.csv", "Calendar date at daily grain", "date", "generated"),
        ("item_id", "inventory_decision_daily.csv", "Stable item identifier", "string", "manually constructed"),
        ("ed_arrivals", "inventory_decision_daily.csv", "Generated ED arrivals observed for the day", "integer", "generated"),
        ("forecast_arrivals_1d", "inventory_decision_daily.csv", "One-day-ahead arrival forecast using prior observations only", "float", "derived"),
        ("forecast_arrivals_7d", "inventory_decision_daily.csv", "Sum of seven daily arrival forecasts", "float", "derived"),
        ("item_demand_requested", "inventory_decision_daily.csv", "Generated requested units for item and day", "integer", "generated"),
        ("item_issued_qty", "inventory_decision_daily.csv", "Units fulfilled from available inventory", "integer", "derived"),
        ("backorder_qty", "inventory_decision_daily.csv", "Unfilled units carried after daily issue", "integer", "derived"),
        ("opening_on_hand", "inventory_decision_daily.csv", "Usable units at start of day", "integer", "derived"),
        ("receipts_qty", "inventory_decision_daily.csv", "Units received on the day", "integer", "generated/derived"),
        ("closing_on_hand", "inventory_decision_daily.csv", "Opening plus receipts minus issued units", "integer", "derived"),
        ("open_order_qty", "inventory_decision_daily.csv", "Units scheduled for future receipt", "integer", "derived"),
        ("inventory_position", "inventory_decision_daily.csv", "Closing stock plus open orders minus backorders", "integer", "derived"),
        ("forecast_item_demand_7d", "inventory_decision_daily.csv", "Seven-day arrivals forecast times item use rate", "float", "derived"),
        ("days_of_supply", "inventory_decision_daily.csv", "Inventory position divided by forecast average daily item demand", "float", "derived"),
        ("stockout_risk_proxy", "inventory_decision_daily.csv", "Logistic shortage-risk proxy for the protection period", "float", "derived"),
        ("recommended_order_qty", "inventory_decision_daily.csv", "Target-stock gap rounded to a full case pack", "integer", "derived"),
        ("recommended_action", "inventory_decision_daily.csv", "Escalate, order, or monitor rule output", "string", "derived"),
        ("order_placed_qty", "inventory_decision_daily.csv", "Units actually ordered that day (review days, or any day risk >= 0.50)", "integer", "derived"),
        ("supplier_event", "inventory_decision_daily.csv", "none, delayed, short_shipped, or both for receipts due that day", "string", "generated"),
        ("lead_time_days", "inventory_decision_daily.csv", "Assumed supplier lead time", "integer", "manually constructed assumption"),
        ("case_pack", "inventory_decision_daily.csv", "Assumed order pack multiple", "integer", "manually constructed assumption"),
        ("unit_cost", "inventory_decision_daily.csv", "Illustrative cost per unit", "float", "manually constructed assumption"),
        ("estimated_order_cost", "inventory_decision_daily.csv", "Recommended quantity times unit cost", "float", "derived"),
        ("use_rate_per_arrival", "inventory_decision_daily.csv", "Assumed expected units used per ED arrival", "float", "manually constructed assumption"),
        ("service_level_target", "inventory_decision_daily.csv", "Assumed target item availability", "float", "manually constructed assumption"),
        ("assumption_version", "all final files", "Version label for active prototype assumptions", "string", "manually constructed"),
        ("record_status", "inventory_decision_daily.csv", "Provenance label for the row", "string", "derived"),
        ("scenario", "scenario_comparison.csv", "Low, base, or high demand case", "string", "manually constructed"),
        ("demand_multiplier", "scenario_comparison.csv", "Scenario multiplier applied to forecast demand", "float", "manually constructed assumption"),
    ]
    write_csv(docs_dir / "data_dictionary.csv", [
        {"field_name": a, "dataset": b, "definition": c, "data_type": d, "provenance": e}
        for a, b, c, d, e in dictionary
    ])

    total_requested = sum(r["item_demand_requested"] for r in final_rows)
    total_issued = sum(min(r["item_issued_qty"], r["item_demand_requested"]) for r in final_rows)
    wape_num = sum(abs(r["ed_arrivals"] - r["forecast_arrivals_1d"]) for r in final_rows)
    wape_den = sum(r["ed_arrivals"] for r in final_rows)
    summary = {
        "random_seed": SEED, "coverage_start": START.isoformat(), "coverage_end": END.isoformat(),
        "arrival_rows": len(arrivals), "item_master_rows": len(item_rows),
        "final_item_day_rows": len(final_rows), "final_item_day_columns": len(final_rows[0]),
        "scenario_rows": len(scenario_rows), "scenario_columns": len(scenario_rows[0]),
        "validation_checks": len(checks), "validation_passed": sum(c["status"] == "PASS" for c in checks),
        "arrival_forecast_wape": round(wape_num / wape_den, 4),
        "fill_rate_proxy": round(total_issued / total_requested, 4),
        "high_risk_item_days": sum(r["stockout_risk_proxy"] >= 0.5 for r in final_rows)
    }
    (docs_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(c["status"] != "PASS" for c in checks):
        raise SystemExit("Validation failed; inspect data/final/validation_results.csv")


if __name__ == "__main__":
    main()
