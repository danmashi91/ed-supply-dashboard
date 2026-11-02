"""Train, compare and validate the predictive models.

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
data/modeling/evaluation_predictions.csv
data/modeling/error_analysis.csv
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
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_features import (
    FORECAST_RATIO_LIMITS,
    ARRIVAL_PREDICTORS,
    DEMAND_FORMULA_PREDICTORS,
    ArrivalFormulaModel,
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
    """Learned candidates compared with the no-fitting formula in main()."""
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
        "Monotone-constrained gradient boosting (Week 11 improvement)": (
            Pipeline([("model", HistGradientBoostingRegressor(
                loss="squared_error", max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE,
                # +1 = the prediction may only rise when this predictor rises.
                monotonic_cst=[1 if c in ARRIVAL_PREDICTORS + ["expected_demand_7d", "demand_lag1", "demand_roll7",
                                                               "use_rate_per_arrival", "surge_flag"] else 0
                               for c in ["expected_demand_7d"] + DEMAND_MODEL_NUMERIC]))]),
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


def date_ordered_splits(data: pd.DataFrame, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV indices based on dates, never on physical row order.

    Rows are stored by item and date, so ``TimeSeriesSplit`` on row positions
    would mix calendar periods across items.  These explicit indices guarantee
    that every calibration fold trains only on dates before its validation dates.
    """
    dates = pd.Series(pd.to_datetime(data["date"]).dt.normalize().unique()).sort_values().to_numpy()
    blocks = np.array_split(dates, n_splits + 1)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    normalized = pd.to_datetime(data["date"]).dt.normalize()
    for i in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:i])
        valid_dates = blocks[i]
        train_idx = np.flatnonzero(normalized.isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(normalized.isin(valid_dates).to_numpy())
        if len(train_idx) and len(valid_idx):
            result.append((train_idx, valid_idx))
    return result


def calibrated(base: Pipeline, data: pd.DataFrame) -> CalibratedClassifierCV:
    """Sigmoid calibration with explicit date-ordered folds.

    Class weighting helps the model rank rare shortages, but it inflates the raw
    scores far above the real shortage rate. Calibration maps them back.
    """
    return CalibratedClassifierCV(base, method="sigmoid", cv=date_ordered_splits(data))


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


def input_ranges(df: pd.DataFrame, raw: list[str]) -> dict:
    """Observed development-data ranges for numeric scenario checks."""
    result = {}
    for col in raw:
        if pd.api.types.is_numeric_dtype(df[col]):
            result[col] = {
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
    return result


def demand_segment_rows(test: pd.DataFrame, pred: np.ndarray, benchmark: np.ndarray) -> pd.DataFrame:
    work = test[["item_id", "surge_flag", "target_demand_next_7d"]].copy()
    work["prediction"] = pred
    work["benchmark"] = benchmark
    work["abs_error"] = np.abs(work["target_demand_next_7d"] - work["prediction"])
    work["signed_error"] = work["prediction"] - work["target_demand_next_7d"]
    work["demand_band"] = pd.qcut(work["target_demand_next_7d"], 4, labels=["lowest", "lower-middle", "upper-middle", "highest"])
    records = []
    for segment_type, col in [("item", "item_id"), ("surge", "surge_flag"), ("demand band", "demand_band")]:
        for value, group in work.groupby(col, observed=False):
            actual = group["target_demand_next_7d"]
            records.append({
                "kpi": "Seven-day demand", "segment_type": segment_type, "segment": str(value),
                "rows": len(group), "positives": "", "mae": round(float(group["abs_error"].mean()), 3),
                "wape": round(wape(actual, group["prediction"].to_numpy()), 4),
                "bias": round(float(group["signed_error"].mean()), 3), "roc_auc": "",
                "average_precision": "", "precision": "", "recall": "",
                "false_positives": "", "false_negatives": "",
            })
    return pd.DataFrame(records)


def shortage_segment_rows(test: pd.DataFrame, prob: np.ndarray, flag: np.ndarray) -> pd.DataFrame:
    work = test[["item_id", "surge_flag", "supplier_event", "target_shortage_next_7d"]].copy()
    work["probability"] = prob
    work["flag"] = flag
    records = []
    for segment_type, col in [("item", "item_id"), ("surge", "surge_flag"), ("supplier event", "supplier_event")]:
        for value, group in work.groupby(col, observed=False):
            y = group["target_shortage_next_7d"].astype(int)
            p = group["probability"]
            f = group["flag"].astype(int)
            has_both = y.nunique() == 2
            records.append({
                "kpi": "Seven-day shortage", "segment_type": segment_type, "segment": str(value),
                "rows": len(group), "positives": int(y.sum()), "mae": "", "wape": "", "bias": "",
                "roc_auc": round(float(roc_auc_score(y, p)), 4) if has_both else "",
                "average_precision": round(float(average_precision_score(y, p)), 4) if y.sum() else "",
                "precision": round(float(precision_score(y, f, zero_division=0)), 4),
                "recall": round(float(recall_score(y, f, zero_division=0)), 4),
                "false_positives": int(((f == 1) & (y == 0)).sum()),
                "false_negatives": int(((f == 0) & (y == 1)).sum()),
            })
    return pd.DataFrame(records)


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
    def direction_violation(predict, v: pd.DataFrame) -> float:
        """Share of rows where 10% more expected patients LOWERS the prediction.

        A what-if tool must not tell a manager that more patients means less
        demand, so candidates that do this on more than 1% of rows are excluded.
        """
        up = v.copy()
        for col in ARRIVAL_PREDICTORS:
            up[col] = up[col] * 1.10
        base = predict(add_model_features(v))
        return float(np.mean(predict(add_model_features(up)) < base - 0.5))

    def evaluate_candidate(make_predictor) -> tuple[list[float], float]:
        scores, violations = [], []
        for start, end in folds:
            f = train[train["date"] < start]
            v = train[(train["date"] >= start) & (train["date"] < end)]
            predict = make_predictor(f)
            scores.append(wape(v[y_d], predict(v)))
            violations.append(direction_violation(predict, v))
        return scores, float(np.mean(violations))

    FORMULA_NAME = "Simple formula: 7-day arrival forecast × units per arrival (no fitting)"
    demand_selection = []
    formula = ArrivalFormulaModel()
    scores, viol = evaluate_candidate(lambda f: formula.predict)
    demand_selection.append({"candidate": FORMULA_NAME, "fold_wape": [round(x, 4) for x in scores],
                             "mean_validation_wape": round(float(np.mean(scores)), 4),
                             "direction_violation_share": round(viol, 4)})
    for name, (pipe, cols) in demand_candidates().items():
        def make(f, p=pipe, c=cols):
            fitted = clone(p).fit(f[c], f[y_d])
            return lambda d: np.maximum(0, fitted.predict(d[c]))
        scores, viol = evaluate_candidate(make)
        demand_selection.append({"candidate": name, "fold_wape": [round(x, 4) for x in scores],
                                 "mean_validation_wape": round(float(np.mean(scores)), 4),
                                 "direction_violation_share": round(viol, 4)})
    # Rule: among candidates that respond in the right direction to patient
    # volume (violations below 1% of rows), keep the lowest mean validation WAPE.
    eligible = [r for r in demand_selection if r["direction_violation_share"] < 0.01]
    demand_choice = min(eligible, key=lambda r: r["mean_validation_wape"])["candidate"]
    if demand_choice == FORMULA_NAME:
        demand_model, demand_cols, demand_raw = formula, DEMAND_FORMULA_PREDICTORS, DEMAND_FORMULA_PREDICTORS
        demand_pred = formula.predict(test)
    else:
        demand_model, demand_cols = demand_candidates()[demand_choice]
        demand_model.fit(train[demand_cols], train[y_d])
        demand_raw = DEMAND_RAW
        demand_pred = np.maximum(0, demand_model.predict(test[demand_cols]))
    # Best learned model, kept for reference in the comparison table and saved file.
    learned = [r for r in demand_selection if r["candidate"] != FORMULA_NAME]
    learned_name = min(learned, key=lambda r: r["mean_validation_wape"])["candidate"]
    demand_benchmark = test["expected_demand_7d"].to_numpy()
    demand_metrics = {
        "wape": round(wape(test[y_d], demand_pred), 4),
        "benchmark_wape": round(wape(test[y_d], demand_benchmark), 4),
        "mae_units": round(float(mean_absolute_error(test[y_d], demand_pred)), 3),
        "rmse_units": round(float(mean_squared_error(test[y_d], demand_pred) ** 0.5), 3),
        "r2": round(float(r2_score(test[y_d], demand_pred)), 4),
    }
    coefficients = {}
    if isinstance(demand_model, Pipeline) and isinstance(demand_model.named_steps["model"], LinearRegression):
        scaler = demand_model.named_steps["prep"].named_transformers_["numeric"]
        raw_coef = demand_model.named_steps["model"].coef_ / scaler.scale_
        coefficients = {c: round(float(v), 4) for c, v in zip(DEMAND_MODEL_NUMERIC, raw_coef)}

    # Final 2025 reporting for every version in the comparison table. This runs
    # after the choice above is fixed, so the test set does not influence it.
    by_name = {r["candidate"]: r for r in demand_selection}
    versions = [
        ("Simple baseline", FORMULA_NAME),
        ("Original Week 10", "Gradient boosting with item code (first Week 10 version)"),
        ("Week 10 revision", "Gradient boosting without item code"),
        ("Week 11 improvement attempt", "Monotone-constrained gradient boosting (Week 11 improvement)"),
    ]
    demand_comparison = []
    saved_learned = None
    for label, name in versions:
        if name == FORMULA_NAME:
            pred = formula.predict(test)
        else:
            pipe, cols = demand_candidates()[name]
            pipe.fit(train[cols], train[y_d])
            pred = np.maximum(0, pipe.predict(test[cols]))
            if name == learned_name:
                saved_learned = pipe
        demand_comparison.append({
            "version": label, "model": name,
            "validation_wape": by_name[name]["mean_validation_wape"],
            "direction_violation_share": by_name[name]["direction_violation_share"],
            "test_wape": round(wape(test[y_d], pred), 4),
            "test_mae_units": round(float(mean_absolute_error(test[y_d], pred)), 3),
            "retained": name == demand_choice,
        })

    # ---------------- KPI 2: shortage ----------------
    y_s = "target_shortage_next_7d"
    shortage_selection = []
    for name, (pipe, cols) in shortage_candidates().items():
        candidate_model = calibrated(pipe, fit_part).fit(fit_part[cols], fit_part[y_s])
        candidate_prob = candidate_model.predict_proba(val_part[cols])[:, 1]
        shortage_selection.append({
            "candidate": name,
            "validation_roc_auc": round(float(roc_auc_score(val_part[y_s], candidate_prob)), 4),
            "validation_average_precision": round(float(average_precision_score(val_part[y_s], candidate_prob)), 4),
        })
    shortage_choice = max(shortage_selection, key=lambda r: r["validation_average_precision"])["candidate"]
    base_pipe, shortage_cols = shortage_candidates()[shortage_choice]

    # Screening threshold: chosen on the validation period only (maximum F1,
    # balancing missed shortages against false alerts).
    threshold_model = calibrated(base_pipe, fit_part).fit(fit_part[shortage_cols], fit_part[y_s])
    val_prob = threshold_model.predict_proba(val_part[shortage_cols])[:, 1]
    grid = np.unique(np.round(val_prob, 4))
    f1s = [f1_score(val_part[y_s], (val_prob >= t).astype(int), zero_division=0) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])

    shortage_model = calibrated(shortage_candidates()[shortage_choice][0], train).fit(train[shortage_cols], train[y_s])
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

    original_shortage_name = "Logistic regression on original predictors"
    original_shortage_pipe, original_shortage_cols = shortage_candidates()[original_shortage_name]
    original_threshold_model = calibrated(original_shortage_pipe, fit_part).fit(
        fit_part[original_shortage_cols], fit_part[y_s]
    )
    original_val_prob = original_threshold_model.predict_proba(val_part[original_shortage_cols])[:, 1]
    original_grid = np.unique(np.round(original_val_prob, 4))
    original_f1s = [f1_score(val_part[y_s], (original_val_prob >= t).astype(int), zero_division=0) for t in original_grid]
    original_threshold = float(original_grid[int(np.argmax(original_f1s))])
    original_shortage = calibrated(shortage_candidates()[original_shortage_name][0], train).fit(
        train[original_shortage_cols], train[y_s]
    )
    original_prob = original_shortage.predict_proba(test[original_shortage_cols])[:, 1]
    original_flag = (original_prob >= original_threshold).astype(int)
    shortage_validation_by_name = {r["candidate"]: r for r in shortage_selection}
    shortage_comparison = [
        {
            "version": "Constant-rate baseline", "model": "2024 shortage prevalence for every row",
            "validation_average_precision": round(float(val_part[y_s].mean()), 4),
            "test_roc_auc": 0.5, "test_average_precision": round(float(y.mean()), 4),
            "test_brier_score": shortage_metrics["brier_score_base_rate"], "test_precision": 0.0,
            "test_recall": 0.0, "retained": False,
        },
        {
            "version": "Original Week 10", "model": original_shortage_name,
            "validation_average_precision": shortage_validation_by_name[original_shortage_name]["validation_average_precision"],
            "test_roc_auc": round(float(roc_auc_score(y, original_prob)), 4),
            "test_average_precision": round(float(average_precision_score(y, original_prob)), 4),
            "test_brier_score": round(float(brier_score_loss(y, original_prob)), 4),
            "test_precision": round(float(precision_score(y, original_flag, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y, original_flag, zero_division=0)), 4),
            "retained": shortage_choice == original_shortage_name,
        },
        {
            "version": "Revised", "model": shortage_choice + " with date-safe calibration",
            "validation_average_precision": shortage_validation_by_name[shortage_choice]["validation_average_precision"],
            "test_roc_auc": shortage_metrics["roc_auc"],
            "test_average_precision": shortage_metrics["average_precision"],
            "test_brier_score": shortage_metrics["brier_score"],
            "test_precision": shortage_metrics["precision"], "test_recall": shortage_metrics["recall"],
            "retained": True,
        },
    ]

    # ---------------- Overall importance (2025 test set) ----------------
    demand_imp = permutation_importance_raw(
        lambda d: np.maximum(0, demand_model.predict(d[demand_cols])), test, demand_raw,
        lambda p: -mean_absolute_error(test[y_d], p), RANDOM_STATE)
    demand_imp.insert(0, "model", "Seven-day demand")
    shortage_imp = permutation_importance_raw(
        lambda d: shortage_model.predict_proba(d[shortage_cols])[:, 1], test, SHORTAGE_RAW,
        lambda p: roc_auc_score(y, p), RANDOM_STATE)
    shortage_imp.insert(0, "model", "Seven-day shortage")
    importance = pd.concat([demand_imp, shortage_imp]).sort_values(["model", "importance"], ascending=[True, False])

    # ---------------- Error analysis (2025 test set) ----------------
    error_analysis = pd.concat([
        demand_segment_rows(test, demand_pred, demand_benchmark),
        shortage_segment_rows(test, prob, flag),
    ], ignore_index=True)
    predictions = test[["date", "item_id", y_d, y_s, "surge_flag", "supplier_event"]].copy()
    predictions["demand_prediction"] = np.round(demand_pred, 4)
    predictions["demand_benchmark"] = np.round(demand_benchmark, 4)
    predictions["demand_error"] = np.round(demand_pred - test[y_d].to_numpy(), 4)
    predictions["shortage_probability"] = np.round(prob, 6)
    predictions["shortage_flag"] = flag

    # ---------------- Save ----------------
    keep = sorted(set(["date", "item_id", y_d, y_s] + DEMAND_RAW + SHORTAGE_RAW))
    df[keep].to_csv(MODELING_DIR / "model_dataset.csv", index=False)
    importance.to_csv(MODELING_DIR / "global_importance.csv", index=False)
    error_analysis.to_csv(MODELING_DIR / "error_analysis.csv", index=False)
    predictions.to_csv(MODELING_DIR / "evaluation_predictions.csv", index=False)
    # The retained demand method is the formula (no fitted file needed). The best
    # learned candidate is saved for reference and reproducibility.
    old_file = MODEL_DIR / "demand_model.joblib"
    if old_file.exists():
        old_file.unlink()
    if saved_learned is None:
        pipe, cols = demand_candidates()[learned_name]
        saved_learned = pipe.fit(train[cols], train[y_d])
    joblib.dump(saved_learned, MODEL_DIR / "demand_learned_model.joblib")
    joblib.dump(shortage_model, MODEL_DIR / "shortage_model.joblib")

    metadata = {
        "version": "v3.1-validation-2026-11-02",
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
        "final_test_policy": "The 2025 test set is used once for final reporting, not to select candidates, thresholds, predictors, or settings.",
        "leakage_checks": [
            "Targets use days t+1 through t+7; predictors are values available by the end of day t.",
            "Lagged demand and the 28-day arrival average use shift(1), so the current or future target window is excluded.",
            "Arrival forecasts are generated only from arrival history available through day t.",
            "Candidate selection and shortage threshold selection use 2024 only; 2025 is final evaluation only.",
            "Probability calibration uses explicit expanding date folds, preventing calendar leakage across item groups.",
        ],
        "demand": {
            "kpi": "Predicted units requested during the next seven days",
            "algorithm": demand_choice,
            "retained_type": "formula" if demand_choice == FORMULA_NAME else "learned",
            "best_learned_candidate": learned_name,
            "selection": demand_selection,
            "selection_rule": ("candidates must respond in the right direction to patient volume (10% more expected "
                               "patients lowers the prediction on under 1% of validation rows); among those, the lowest "
                               "mean WAPE over four rolling 2024 validation folds is retained"),
            "raw_predictors": demand_raw,
            "model_predictors": demand_cols,
            "coefficients_per_unit": coefficients,
            "metrics": demand_metrics,
            "reference_values": reference_values(train, demand_raw),
            "input_ranges": input_ranges(train, DEMAND_RAW),
            "forecast_ratio_limits": list(FORECAST_RATIO_LIMITS),
            "original_vs_revised": demand_comparison,
            "improvement_conclusion": (
                "Week 11 testing found that the Week 10 gradient boosting model lowered its demand prediction when more "
                f"patients were expected on {by_name['Gradient boosting without item code']['direction_violation_share']:.0%} "
                "of validation rows, so its what-if results were not credible. A monotone-constrained model fixed the "
                f"direction ({by_name['Monotone-constrained gradient boosting (Week 11 improvement)']['mean_validation_wape']:.2%} "
                "validation WAPE) but no learned model beat the simple formula "
                f"({by_name[FORMULA_NAME]['mean_validation_wape']:.2%}). The formula is retained as the KPI 1 prediction; "
                "the learned models are kept as documented comparisons."
                if demand_choice == FORMULA_NAME else
                f"{demand_choice} is retained under the selection rule."
            ),
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
            "input_ranges": input_ranges(train, SHORTAGE_RAW),
            "input_ranges_by_item": {
                item: input_ranges(group, ["closing_on_hand"]) for item, group in train.groupby("item_id")
            },
            "original_vs_revised": shortage_comparison,
            "improvement_conclusion": "Coverage and trend predictors improved rare-event ranking over the original raw predictor set. Date-safe calibration fixes the internal temporal leakage risk. The revised model is retained as a screening model.",
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
            "Performance by item and supplier-event subgroup is uncertain because several segments contain very few or no shortage cases.",
        ],
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"demand_choice": demand_choice, "demand_selection": demand_selection,
                      "demand": demand_metrics, "coefficients": coefficients,
                      "shortage_choice": shortage_choice, "shortage_selection": shortage_selection,
                      "shortage": shortage_metrics,
                      "demand_comparison": demand_comparison,
                      "shortage_comparison": shortage_comparison}, indent=2))
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
