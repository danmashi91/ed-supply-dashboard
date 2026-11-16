#!/usr/bin/env python3
"""Compare the retained KPI models with simple, decision-relevant baselines on the 2025 test period.

Outputs evidence/outputs/baseline_comparison.csv.
Demand baselines: (a) persistence (the previous 7 days' demand), (b) the retained arrival formula.
Shortage baselines: (a) constant rate, (b) rank by fewest days of supply, (c) rule 'days_of_supply < lead time',
(d) rank by the descriptive stockout_risk_proxy; each compared with the model at the SAME alert share.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data"
ev = pd.read_csv(D / "modeling/evaluation_predictions.csv", parse_dates=["date"])
md = pd.read_csv(D / "modeling/model_dataset.csv", parse_dates=["date"])
daily = pd.read_csv(D / "final/inventory_decision_daily.csv", parse_dates=["date"]).sort_values(["item_id", "date"])
meta = json.loads((ROOT / "models/model_metadata.json").read_text())
rows = []
# ---- demand
daily["persist"] = daily.groupby("item_id").item_demand_requested.transform(lambda s: s.rolling(7).sum())
t = ev.merge(daily[["date", "item_id", "persist"]], on=["date", "item_id"])
def wape(y, p): return float((y - p).abs().sum() / y.sum())
def mae(y, p): return float((y - p).abs().mean())
surge = t.surge_flag == 1
for name, col in [("Persistence baseline (previous 7 days' demand)", "persist"), ("Retained arrival formula", "demand_prediction")]:
    rows.append({"kpi": "7-day demand", "method": name, "metric": "WAPE all test days", "value": round(wape(t.target_demand_next_7d, t[col]), 4)})
    rows.append({"kpi": "7-day demand", "method": name, "metric": "WAPE surge days", "value": round(wape(t.target_demand_next_7d[surge], t[col][surge]), 4)})
    rows.append({"kpi": "7-day demand", "method": name, "metric": "MAE units", "value": round(mae(t.target_demand_next_7d, t[col]), 2)})
    rows.append({"kpi": "7-day demand", "method": name, "metric": "mean signed error on surge days (negative = under-prediction), units", "value": round(float((t[col][surge] - t.target_demand_next_7d[surge]).mean()), 1)})
# ---- shortage
m = md[["date", "item_id", "days_of_supply", "lead_time_days"]].merge(daily[["date", "item_id", "stockout_risk_proxy"]], on=["date", "item_id"])
s = ev.merge(m, on=["date", "item_id"]); y = s.target_shortage_next_7d.values
alert_share = float(s.shortage_flag.mean()); k = int(round(alert_share * len(s)))
def topk_recall(score):
    idx = np.argsort(-score, kind="stable")[:k]; hit = y[idx].sum(); return float(hit / y.sum()), float(hit / k)
cands = {"Constant rate (2024 prevalence)": np.full(len(s), meta["shortage"]["metrics"]["observed_rate"]),
         "Rank by fewest days of supply": -s.days_of_supply.values,
         "Rank by descriptive stockout_risk_proxy": s.stockout_risk_proxy.values,
         "Retained shortage model (calibrated logistic regression)": s.shortage_probability.values}
for name, score in cands.items():
    rec_, prec = topk_recall(score)
    rows.append({"kpi": "7-day shortage", "method": name, "metric": f"ROC-AUC", "value": round(float(roc_auc_score(y, score)), 4)})
    rows.append({"kpi": "7-day shortage", "method": name, "metric": "average precision", "value": round(float(average_precision_score(y, score)), 4)})
    rows.append({"kpi": "7-day shortage", "method": name, "metric": f"recall when flagging the top {alert_share:.2%} of item-days", "value": round(rec_, 4)})
    rows.append({"kpi": "7-day shortage", "method": name, "metric": f"precision when flagging the top {alert_share:.2%} of item-days", "value": round(prec, 4)})
rule = (s.days_of_supply < s.lead_time_days).values
rows.append({"kpi": "7-day shortage", "method": "Rule: days of supply < lead time", "metric": "share of item-days flagged", "value": round(float(rule.mean()), 4)})
rows.append({"kpi": "7-day shortage", "method": "Rule: days of supply < lead time", "metric": "recall", "value": round(float(y[rule].sum() / y.sum()), 4)})
rows.append({"kpi": "7-day shortage", "method": "Rule: days of supply < lead time", "metric": "precision", "value": round(float(y[rule].mean()) if rule.any() else 0.0, 4)})
out = pd.DataFrame(rows); out.to_csv(ROOT / "evidence/outputs/baseline_comparison.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90); print(out.to_string(index=False))
