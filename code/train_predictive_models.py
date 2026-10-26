"""Train, select and validate the Week 10 predictive models.

The models use only synthetic project data. Time order is preserved at every
step, so no model is trained on dates later than the dates it is judged on:

1. Model selection: candidates are fitted on 2024-01-01 to 2024-09-30 and
   compared on 2024-10-01 to 2024-12-31 (the validation period).
2. Final fit: the chosen model is refitted on all of 2024.
3. Final test: the refitted model is scored once on 2025 (the test period).

KPI 1 — seven-day item demand (regression).
KPI 2 — seven-day shortage probability (classification, calibrated).

Run from the project root:  python code/train_predictive_models.py

Outputs
-------
models/demand_model.joblib
models/shortage_model.joblib
models/model_metadata.json
data/modeling/model_dataset.csv
data/modeling/global_importance.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_features import (
    DEMAND_MODEL_NUMERIC,
    DEMAND_RAW,
    SHORTAGE_MODEL_CATEGORICAL,
    SHORTAGE_MODEL_NUMERIC,
    SHORTAGE_RAW,
    add_model_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODELING_DIR = DATA / "modeling"
VALIDATION_START = pd.Timestamp("2024-10-01")
SPLIT_DATE = pd.Timestamp("2025-01-01")
RANDOM_STATE = 20261024
N_PERMUTATION_REPEATS = 8

# Predictors used by the first Week 10 candidates (kept for comparison).
ORIGINAL_DEMAND_NUMERIC = [
    "ed_arrivals", "forecast_arrivals_1d", "forecast_arrivals_7d", "demand_lag1", "demand_roll7",
    "use_rate_per_arrival", "day_of_week_num", "month", "weekend_flag", "holiday_flag", "surge_flag",
]
ORIGINAL_SHORTAGE_NUMERIC = [
    "forecast_arrivals_7d", "demand_lag1", "demand_roll7", "inventory_position", "closing_on_hand",
    "open_order_qty", "backorder_qty", "days_of_supply", "lead_time_days", "use_rate_per_arrival",
    "service_level_target", "surge_flag",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def future_sum(series: pd.Series, horizon: int = 7) -> pd.Series:
    """Sum values from t+1 through t+horizon within one item."""
    return sum(series.shift(-step) for step in range(1, horizon + 1))


def future_any_positive(series: pd.Series, horizon: int = 7) -> pd.Series:
    """Return 1 when any value from t+1 through t+horizon is positive."""
    shifted = pd.concat([series.shift(-step).gt(0) for step in range(1, horizon + 1)], axis=1)
    return shifted.any(axis=1).astype(int)


def build_model_dataset() -> pd.DataFrame:
    """One row per item-day with predictors known on day t and targets for t+1..t+7."""
    daily = pd.read_csv(DATA / "final" / "inventory_decision_daily.csv", parse_dates=["date"])
    arrivals = pd.read_csv(DATA / "raw" / "generated_ed_arrivals.csv", parse_dates=["date"])
    arrivals = arrivals.sort_values("date")
    # Average daily arrivals over the 28 days before day t (uses past days only).
    arrivals["avg_arrivals_28d"] = arrivals["ed_arrivals"].shift(1).rolling(28, min_periods=1).mean()
    arrivals["avg_arrivals_28d"] = arrivals["avg_arrivals_28d"].fillna(arrivals["ed_arrivals"])
    flags = arrivals[["date", "weekend_flag", "holiday_flag", "surge_flag", "avg_arrivals_28d"]]
    df = daily.merge(flags, on="date", how="left", validate="many_to_one")
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    df["day_of_week_num"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["demand_lag1"] = df.groupby("item_id")["item_demand_requested"].shift(1)
    df["demand_roll7"] = df.groupby("item_id")["item_demand_requested"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["target_demand_next_7d"] = df.groupby("item_id")["item_demand_requested"].transform(future_sum)
    df["target_shortage_next_7d"] = df.groupby("item_id")["backorder_qty"].transform(future_any_positive)
    # The final seven days have incomplete forward outcomes.
    df = df[df["date"] <= df["date"].max() - pd.Timedelta(days=7)].copy()
    df["demand_lag1"] = df["demand_lag1"].fillna(df["item_demand_requested"])
    df["demand_roll7"] = df["demand_roll7"].fillna(df["item_demand_requested"])
    return add_model_features(df)


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    parts = [("numeric", StandardScaler(), numeric)]
    if categorical:
        parts.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def wape(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / actual.sum())


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def gradient_boosting() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=180,
                                     learning_rate=0.04, max_depth=3, loss="huber")


def demand_candidates() -> dict[str, tuple[Pipeline, list[str]]]:
    """Listed in order of preference; see TIE_MARGIN in main()."""
    return {
        "Gradient boosting without item code": (
            Pipeline([("prep", preprocessor(DEMAND_MODEL_NUMERIC, [])), ("model", gradient_boosting())]),
            DEMAND_MODEL_NUMERIC,
        ),
        "Gradient boosting with item code (first Week 10 version)": (
            Pipeline([("prep", preprocessor(ORIGINAL_DEMAND_NUMERIC, ["item_id"])), ("model", gradient_boosting())]),
            ORIGINAL_DEMAND_NUMERIC + ["item_id"],
        ),
        "Gradient boosting with arrival-driven expected demand": (
            Pipeline([("prep", preprocessor(["expected_demand_7d"] + DEMAND_MODEL_NUMERIC, [])),
                      ("model", gradient_boosting())]),
            ["expected_demand_7d"] + DEMAND_MODEL_NUMERIC,
        ),
        "Linear regression on expected and recent demand": (
            Pipeline([("prep", preprocessor(["expected_demand_7d", "recent_demand_7d"], [])),
                      ("model", LinearRegression())]),
            ["expected_demand_7d", "recent_demand_7d"],
        ),
    }


def logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor(numeric, categorical)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
    ])


def shortage_candidates() -> dict[str, tuple[Pipeline, list[str]]]:
    model_cols = SHORTAGE_MODEL_NUMERIC + SHORTAGE_MODEL_CATEGORICAL
    return {
        "Logistic regression on coverage and trend predictors": (
            logistic(SHORTAGE_MODEL_NUMERIC, SHORTAGE_MODEL_CATEGORICAL), model_cols),
        "Logistic regression on original predictors": (
            logistic(ORIGINAL_SHORTAGE_NUMERIC, ["item_id", "supplier_event"]),
            ORIGINAL_SHORTAGE_NUMERIC + ["item_id", "supplier_event"]),
    }


def calibrated(base: Pipeline) -> CalibratedClassifierCV:
    """Sigmoid calibration with time-ordered folds, so the output is a usable probability.

    Class weighting helps the model rank rare shortages, but it inflates the raw
    scores far above the real shortage rate. Calibration maps them back.
    """
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=4))


# ---------------------------------------------------------------------------
# Overall importance on raw (user-facing) predictors
# ---------------------------------------------------------------------------
def permutation_importance_raw(predict, data: pd.DataFrame, raw: list[str], score, seed: int) -> pd.DataFrame:
    """Shuffle one raw predictor at a time, recompute derived predictors, and
    record how much the test score gets worse. Higher = the model relies on it more."""
    rng = np.random.default_rng(seed)
    base_score = score(predict(add_model_features(data)))
    rows = []
    for col in raw:
        drops = []
        for _ in range(N_PERMUTATION_REPEATS):
            shuffled = data.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base_score - score(predict(add_model_features(shuffled))))
        rows.append({"predictor": col, "importance": max(0.0, float(np.mean(drops))),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows)


def reference_values(df: pd.DataFrame, raw: list[str]) -> dict:
    """Typical training value for each raw predictor (median or most common)."""
    result: dict[str, float | int | str] = {}
    for col in raw:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            result[col] = str(df[col].mode().iloc[0])
        else:
            value = float(df[col].median())
            result[col] = int(value) if value.is_integer() else round(value, 4)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    df = build_model_dataset()
    train = df[df["date"] < SPLIT_DATE].copy()
    test = df[df["date"] >= SPLIT_DATE].copy()
    fit_part = train[train["date"] < VALIDATION_START]
    val_part = train[train["date"] >= VALIDATION_START]

    # ---------------- KPI 1: demand ----------------
    # Rolling-origin validation inside 2024: each fold trains on all earlier
    # dates and validates on the next two months.
    y_d = "target_demand_next_7d"
    folds = [("2024-05-01", "2024-07-01"), ("2024-07-01", "2024-09-01"),
             ("2024-09-01", "2024-11-01"), ("2024-11-01", "2025-01-01")]
    def fold_scores(fit_and_predict) -> list[float]:
        scores = []
        for start, end in folds:
            f = train[train["date"] < start]
            v = train[(train["date"] >= start) & (train["date"] < end)]
            scores.append(wape(v[y_d], fit_and_predict(f, v)))
        return scores
    demand_selection = []
    bench = fold_scores(lambda f, v: v["expected_demand_7d"].to_numpy())
    demand_selection.append({"candidate": "Benchmark: 7-day arrival forecast × units per arrival (no fitting)",
                             "fold_wape": [round(x, 4) for x in bench], "mean_validation_wape": round(float(np.mean(bench)), 4)})
    for name, (pipe, cols) in demand_candidates().items():
        scores = fold_scores(lambda f, v, p=pipe, c=cols: np.maximum(0, clone(p).fit(f[c], f[y_d]).predict(v[c])))
        demand_selection.append({"candidate": name, "fold_wape": [round(x, 4) for x in scores],
                                 "mean_validation_wape": round(float(np.mean(scores)), 4)})
    # Fitted candidates within TIE_MARGIN of the best are treated as tied; the
    # first-listed (preferred) design wins a tie.
    TIE_MARGIN = 0.0025
    fitted = [r for r in demand_selection if not r["candidate"].startswith("Benchmark")]
    best = min(r["mean_validation_wape"] for r in fitted)
    demand_choice = next(r["candidate"] for r in fitted if r["mean_validation_wape"] <= best + TIE_MARGIN)
    demand_model, demand_cols = demand_candidates()[demand_choice]
    demand_model.fit(train[demand_cols], train[y_d])
    demand_pred = np.maximum(0, demand_model.predict(test[demand_cols]))
    demand_metrics = {
        "wape": round(wape(test[y_d], demand_pred), 4),
        "benchmark_wape": round(wape(test[y_d], test["expected_demand_7d"].to_numpy()), 4),
        "mae_units": round(float(mean_absolute_error(test[y_d], demand_pred)), 3),
        "rmse_units": round(float(mean_squared_error(test[y_d], demand_pred) ** 0.5), 3),
        "r2": round(float(r2_score(test[y_d], demand_pred)), 4),
    }
    coefficients = {}
    if isinstance(demand_model.named_steps["model"], LinearRegression):
        scaler = demand_model.named_steps["prep"].named_transformers_["numeric"]
        raw_coef = demand_model.named_steps["model"].coef_ / scaler.scale_
        coefficients = {c: round(float(v), 4) for c, v in zip(DEMAND_MODEL_NUMERIC, raw_coef)}

    # ---------------- KPI 2: shortage ----------------
    y_s = "target_shortage_next_7d"
    shortage_selection = []
    for name, (pipe, cols) in shortage_candidates().items():
        pipe.fit(fit_part[cols], fit_part[y_s])
        shortage_selection.append({
            "candidate": name,
            "validation_roc_auc": round(float(roc_auc_score(val_part[y_s], pipe.predict_proba(val_part[cols])[:, 1])), 4),
            "validation_average_precision": round(float(average_precision_score(
                val_part[y_s], pipe.predict_proba(val_part[cols])[:, 1])), 4),
        })
    shortage_choice = max(shortage_selection, key=lambda r: r["validation_average_precision"])["candidate"]
    base_pipe, shortage_cols = shortage_candidates()[shortage_choice]

    # Screening threshold: chosen on the validation period only (maximum F1,
    # balancing missed shortages against false alerts).
    threshold_model = calibrated(base_pipe).fit(fit_part[shortage_cols], fit_part[y_s])
    val_prob = threshold_model.predict_proba(val_part[shortage_cols])[:, 1]
    grid = np.unique(np.round(val_prob, 4))
    f1s = [f1_score(val_part[y_s], (val_prob >= t).astype(int), zero_division=0) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])

    shortage_model = calibrated(shortage_candidates()[shortage_choice][0]).fit(train[shortage_cols], train[y_s])
    prob = shortage_model.predict_proba(test[shortage_cols])[:, 1]
    flag = (prob >= threshold).astype(int)
    y = test[y_s].astype(int)
    shortage_metrics = {
        "positive_cases": int(y.sum()),
        "test_rows": int(len(y)),
        "observed_rate": round(float(y.mean()), 4),
        "mean_predicted_probability": round(float(prob.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "average_precision": round(float(average_precision_score(y, prob)), 4),
        "brier_score": round(float(brier_score_loss(y, prob)), 4),
        "brier_score_base_rate": round(float(brier_score_loss(y, np.full(len(y), train[y_s].mean()))), 4),
        "screening_threshold": round(threshold, 4),
        "alerts": int(flag.sum()),
        "alert_share": round(float(flag.mean()), 4),
        "precision": round(float(precision_score(y, flag, zero_division=0)), 4),
        "recall": round(float(recall_score(y, flag, zero_division=0)), 4),
        "f1": round(float(f1_score(y, flag, zero_division=0)), 4),
    }

    # ---------------- Overall importance (2025 test set) ----------------
    demand_imp = permutation_importance_raw(
        lambda d: np.maximum(0, demand_model.predict(d[demand_cols])), test, DEMAND_RAW,
        lambda p: -mean_absolute_error(test[y_d], p), RANDOM_STATE)
    demand_imp.insert(0, "model", "Seven-day demand")
    shortage_imp = permutation_importance_raw(
        lambda d: shortage_model.predict_proba(d[shortage_cols])[:, 1], test, SHORTAGE_RAW,
        lambda p: roc_auc_score(y, p), RANDOM_STATE)
    shortage_imp.insert(0, "model", "Seven-day shortage")
    importance = pd.concat([demand_imp, shortage_imp]).sort_values(["model", "importance"], ascending=[True, False])

    # ---------------- Save ----------------
    keep = sorted(set(["date", "item_id", y_d, y_s] + DEMAND_RAW + SHORTAGE_RAW))
    df[keep].to_csv(MODELING_DIR / "model_dataset.csv", index=False)
    importance.to_csv(MODELING_DIR / "global_importance.csv", index=False)
    joblib.dump(demand_model, MODEL_DIR / "demand_model.joblib")
    joblib.dump(shortage_model, MODEL_DIR / "shortage_model.joblib")

    metadata = {
        "version": "v2.1-week10-2026-10-25",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "training_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "selection_period": {
            "fit": [str(fit_part["date"].min().date()), str(fit_part["date"].max().date())],
            "validation": [str(val_part["date"].min().date()), str(val_part["date"].max().date())],
        },
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "split_method": "chronological; candidates compared only on 2024 data (rolling folds for demand, Oct-Dec 2024 for shortage), final models refitted on all of 2024 and tested once on 2025",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "demand": {
            "kpi": "Predicted units requested during the next seven days",
            "algorithm": demand_choice,
            "selection": demand_selection,
            "selection_rule": "lowest mean WAPE over four rolling 2024 validation folds; candidates within 0.25 percentage points are treated as tied and the first-listed (simplest, no item code) design is preferred",
            "raw_predictors": DEMAND_RAW,
            "model_predictors": demand_cols,
            "coefficients_per_unit": coefficients,
            "metrics": demand_metrics,
            "reference_values": reference_values(train, DEMAND_RAW),
        },
        "shortage": {
            "kpi": "Probability of any backorder during the next seven days",
            "algorithm": shortage_choice + ", class-weighted, with sigmoid probability calibration",
            "selection": shortage_selection,
            "selection_rule": "highest validation average precision (suited to rare outcomes)",
            "raw_predictors": SHORTAGE_RAW,
            "model_predictors": shortage_cols,
            "metrics": shortage_metrics,
            "reference_values": reference_values(train, SHORTAGE_RAW),
            "decision_threshold": round(threshold, 4),
            "threshold_rule": "maximum F1 on the Oct-Dec 2024 validation period",
        },
        "interpretation_method": (
            "Overall importance uses test-set permutation importance on the raw predictors. Individual "
            "contributions use one-predictor-at-a-time replacement with the training reference value; "
            "contributions are predictive, approximate, may not sum exactly because of interactions, "
            "and are not causal effects."
        ),
        "limitations": [
            "All observations and outcomes are synthetic.",
            "Only eight consumables from one simulated emergency department are represented.",
            "Shortages are rare (24 positive test rows), so precision is low and probabilities need recalibration on real data.",
            "Scenario changes are model-based comparisons, not guaranteed intervention effects.",
        ],
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"demand_choice": demand_choice, "demand_selection": demand_selection,
                      "demand": demand_metrics, "coefficients": coefficients,
                      "shortage_choice": shortage_choice, "shortage_selection": shortage_selection,
                      "shortage": shortage_metrics}, indent=2))
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
