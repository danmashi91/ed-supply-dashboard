#!/usr/bin/env python3
"""Sensitivity of the validated KPI models to plausible changes in the synthetic-data generation assumptions.

Usage: python validation/tests/run_sensitivity.py <python-executable> [variant ...]
Each variant copies the repository to a temporary directory, patches generator constants, regenerates the data,
retrains both KPI models with the unchanged selection rules, and records headline test metrics plus a naive
persistence baseline. Results go to evidence/outputs/sensitivity/<variant>.json and sensitivity_summary.csv.
"""
from __future__ import annotations
import json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "outputs" / "sensitivity"

VARIANTS = {
    "V0_baseline": {"desc": "Committed assumptions (reproduction)", "patch": {}},
    "V1_new_seed": {"desc": "Different random seed (20261010)", "patch": {r"SEED = 20261009": "SEED = 20261010"}},
    "V2_supplier_disruptions_x2": {"desc": "Supplier delay and short-ship probabilities doubled (P_DELAY 0.16, P_SHORT 0.10)",
                                   "patch": {r"P_DELAY = 0\.08": "P_DELAY = 0.16", r"P_SHORT = 0\.05": "P_SHORT = 0.10"}},
    "V3_stronger_surges": {"desc": "10 surge episodes (was 6) with lift 1.30-1.60 (was 1.18-1.38)",
                           "patch": {r"N_SURGES = 6": "N_SURGES = 10", r"SURGE_LIFT = \(1\.18, 1\.38\)": "SURGE_LIFT = (1.30, 1.60)"}},
    "V4_lower_volume": {"desc": "Baseline arrivals 150/day instead of 185/day",
                        "patch": {r"return 185\.0 \*": "return 150.0 *"}},
}


def run_variant(name: str, python: str) -> dict:
    spec = VARIANTS[name]
    tmp = Path(tempfile.mkdtemp(prefix=f"sens_{name}_"))
    repo = tmp / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "_apptest_*", "evidence", "__pycache__"))
    gen = repo / "code" / "generate_and_validate.py"
    text = gen.read_text(encoding="utf-8")
    for pat, rep in spec["patch"].items():
        new, n = re.subn(pat, rep, text)
        assert n >= 1, f"patch did not apply: {pat}"
        text = new
    gen.write_text(text, encoding="utf-8")
    t0 = time.time()
    for script in ("generate_and_validate.py", "train_predictive_models.py"):
        r = subprocess.run([python, str(repo / "code" / script)], cwd=repo, capture_output=True, text=True)
        if r.returncode != 0:
            return {"variant": name, "error": r.stderr[-800:]}
    meta = json.loads((repo / "models" / "model_metadata.json").read_text())
    d = pd.read_csv(repo / "data" / "final" / "inventory_decision_daily.csv", parse_dates=["date"]).sort_values(["item_id", "date"])
    tgt = d.groupby("item_id").item_demand_requested.transform(lambda s: s[::-1].rolling(7).sum()[::-1].shift(-1))
    naive = d.groupby("item_id").item_demand_requested.transform(lambda s: s.rolling(7).sum())
    m = (d.date >= "2025-01-01") & tgt.notna() & naive.notna()
    naive_wape = float((tgt[m] - naive[m]).abs().sum() / tgt[m].sum())
    sm, dm = meta["shortage"]["metrics"], meta["demand"]["metrics"]
    orig = {r["version"]: r for r in meta["shortage"]["original_vs_revised"]}
    base_ap = orig.get("Constant-rate baseline", {}).get("test_average_precision")
    res = {
        "variant": name, "description": spec["desc"], "runtime_s": round(time.time() - t0),
        "demand_retained": meta["demand"]["algorithm"][:60], "demand_test_wape": dm["wape"], "naive_persistence_wape": round(naive_wape, 4),
        "demand_direction_violation": meta["demand"]["selection"][0].get("direction_violation_share"),
        "shortage_positive_test_rows": sm["positive_cases"], "shortage_observed_rate": sm["observed_rate"],
        "shortage_roc_auc": sm["roc_auc"], "shortage_avg_precision": sm["average_precision"], "constant_baseline_ap": base_ap,
        "shortage_precision": sm["precision"], "shortage_recall": sm["recall"], "shortage_alert_share": sm["alert_share"],
        "shortage_threshold": sm["screening_threshold"],
    }
    shutil.rmtree(tmp, ignore_errors=True)
    return res


if __name__ == "__main__":
    python = sys.argv[1]
    names = sys.argv[2:] or list(VARIANTS)
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        r = run_variant(n, python)
        (OUT / f"{n}.json").write_text(json.dumps(r, indent=2))
        print(n, json.dumps(r)[:300], flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("V*.json"))]
    pd.DataFrame(rows).to_csv(ROOT / "evidence" / "outputs" / "sensitivity_summary.csv", index=False)
