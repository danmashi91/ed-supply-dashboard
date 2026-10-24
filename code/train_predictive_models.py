"""Train and validate the Week 10 predictive models.

The models use only synthetic project data. A chronological split prevents
future rows from being used to train predictions for earlier dates.

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODELING_DIR = DATA / "modeling"
SPLIT_DATE = pd.Timestamp("2025-01-01")
RANDOM_STATE = 20261024

DEMAND_NUMERIC = [
    "ed_arrivals",
    "forecast_arrivals_1d",
    "forecast_arrivals_7d",
    "demand_lag1",
    "demand_roll7",
    "use_rate_per_arrival",
    "day_of_week_num",
    "month",
    "weekend_flag",
    "holiday_flag",
    "surge_flag",
]
DEMAND_CATEGORICAL = ["item_id"]

SHORTAGE_NUMERIC = [
    "forecast_arrivals_7d",
    "demand_lag1",
    "demand_roll7",
    "inventory_position",
    "closing_on_hand",
    "open_order_qty",
    "backorder_qty",
    "days_of_supply",
    "lead_time_days",
    "use_rate_per_arrival",
    "service_level_target",
    "surge_flag",
]
SHORTAGE_CATEGORICAL = ["item_id", "supplier_event"]


def future_sum(series: pd.Series, horizon: int = 7) -> pd.Series:
    """Sum values from t+1 through t+horizon within one item."""
    return sum(series.shift(-step) for step in range(1, horizon + 1))


def future_any_positive(series: pd.Series, horizon: int = 7) -> pd.Series:
    """Return 1 when any value from t+1 through t+horizon is positive."""
    shifted = pd.concat([series.shift(-step).gt(0) for step in range(1, horizon + 1)], axis=1)
    return shifted.any(axis=1).astype(int)


def build_model_dataset() -> pd.DataFrame:
    daily = pd.read_csv(DATA / "final" / "inventory_decision_daily.csv", parse_dates=["date"])
    arrivals = pd.read_csv(DATA / "raw" / "generated_ed_arrivals.csv", parse_dates=["date"])
    flags = arrivals[["date", "day_of_week", "weekend_flag", "holiday_flag", "surge_flag"]]
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
    # The final seven rows per item have incomplete forward outcomes.
    final_date = df["date"].max()
    df = df[df["date"] <= final_date - pd.Timedelta(days=7)].copy()
    df["demand_lag1"] = df["demand_lag1"].fillna(df["item_demand_requested"])
    df["demand_roll7"] = df["demand_roll7"].fillna(df["item_demand_requested"])
    return df


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def reference_values(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    result: dict[str, float | int | str] = {}
    for col in numeric:
        value = float(df[col].median())
        result[col] = int(value) if float(value).is_integer() else value
    for col in categorical:
        result[col] = str(df[col].mode().iloc[0])
    return result


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    df = build_model_dataset()
    train = df[df["date"] < SPLIT_DATE].copy()
    test = df[df["date"] >= SPLIT_DATE].copy()

    demand_features = DEMAND_NUMERIC + DEMAND_CATEGORICAL
    shortage_features = SHORTAGE_NUMERIC + SHORTAGE_CATEGORICAL

    demand_model = Pipeline(
        [
            ("prep", preprocessor(DEMAND_NUMERIC, DEMAND_CATEGORICAL)),
            ("model", GradientBoostingRegressor(
                random_state=RANDOM_STATE, n_estimators=180, learning_rate=0.04,
                max_depth=3, loss="huber",
            )),
        ]
    )
    demand_model.fit(train[demand_features], train["target_demand_next_7d"])
    demand_pred = np.maximum(0, demand_model.predict(test[demand_features]))

    shortage_model = Pipeline(
        [
            ("prep", preprocessor(SHORTAGE_NUMERIC, SHORTAGE_CATEGORICAL)),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE,
            )),
        ]
    )
    shortage_model.fit(train[shortage_features], train["target_shortage_next_7d"])
    shortage_prob = shortage_model.predict_proba(test[shortage_features])[:, 1]
    shortage_pred = (shortage_prob >= 0.50).astype(int)

    demand_denom = float(test["target_demand_next_7d"].sum())
    demand_metrics = {
        "mae_units": round(float(mean_absolute_error(test["target_demand_next_7d"], demand_pred)), 3),
        "rmse_units": round(float(mean_squared_error(test["target_demand_next_7d"], demand_pred) ** 0.5), 3),
        "wape": round(float(np.abs(test["target_demand_next_7d"] - demand_pred).sum() / demand_denom), 4),
        "r2": round(float(r2_score(test["target_demand_next_7d"], demand_pred)), 4),
    }
    y = test["target_shortage_next_7d"].astype(int)
    shortage_metrics = {
        "positive_cases": int(y.sum()),
        "test_rows": int(len(y)),
        "precision": round(float(precision_score(y, shortage_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, shortage_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, shortage_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, shortage_prob)), 4),
        "brier_score": round(float(brier_score_loss(y, shortage_prob)), 4),
    }

    demand_perm = permutation_importance(
        demand_model, test[demand_features], test["target_demand_next_7d"],
        scoring="neg_mean_absolute_error", n_repeats=8, random_state=RANDOM_STATE,
    )
    shortage_perm = permutation_importance(
        shortage_model, test[shortage_features], y,
        scoring="roc_auc", n_repeats=8, random_state=RANDOM_STATE,
    )
    importance_rows = []
    for feature, value, sd in zip(demand_features, demand_perm.importances_mean, demand_perm.importances_std):
        importance_rows.append({"model": "Seven-day demand", "predictor": feature,
                                "importance": max(0.0, float(value)), "std": float(sd)})
    for feature, value, sd in zip(shortage_features, shortage_perm.importances_mean, shortage_perm.importances_std):
        importance_rows.append({"model": "Seven-day shortage", "predictor": feature,
                                "importance": max(0.0, float(value)), "std": float(sd)})
    importance = pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False])

    output_cols = sorted(set(
        ["date", "item_id", "target_demand_next_7d", "target_shortage_next_7d"]
        + demand_features + shortage_features
    ))
    df[output_cols].to_csv(MODELING_DIR / "model_dataset.csv", index=False)
    importance.to_csv(MODELING_DIR / "global_importance.csv", index=False)
    joblib.dump(demand_model, MODEL_DIR / "demand_model.joblib")
    joblib.dump(shortage_model, MODEL_DIR / "shortage_model.joblib")

    metadata = {
        "version": "v2.0-week10-2026-10-24",
        "random_state": RANDOM_STATE,
        "training_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "split_method": "chronological; 2024 training and 2025 testing",
        "demand": {
            "kpi": "Predicted units requested during the next seven days",
            "algorithm": "GradientBoostingRegressor",
            "features": demand_features,
            "numeric_features": DEMAND_NUMERIC,
            "categorical_features": DEMAND_CATEGORICAL,
            "metrics": demand_metrics,
            "reference_values": reference_values(train, DEMAND_NUMERIC, DEMAND_CATEGORICAL),
        },
        "shortage": {
            "kpi": "Probability of any backorder during the next seven days",
            "algorithm": "Class-weighted LogisticRegression",
            "features": shortage_features,
            "numeric_features": SHORTAGE_NUMERIC,
            "categorical_features": SHORTAGE_CATEGORICAL,
            "metrics": shortage_metrics,
            "reference_values": reference_values(train, SHORTAGE_NUMERIC, SHORTAGE_CATEGORICAL),
            "decision_threshold": 0.50,
        },
        "interpretation_method": (
            "Overall importance uses test-set permutation importance. Individual contributions use "
            "model-agnostic one-feature-at-a-time reference replacement; contributions are predictive, "
            "approximate, may not sum exactly because of interactions, and are not causal effects."
        ),
        "limitations": [
            "All observations and outcomes are synthetic.",
            "Only eight consumables from one simulated emergency department are represented.",
            "The shortage outcome is rare, so probability estimates require calibration with real data.",
            "Scenario changes are model-based comparisons, not guaranteed intervention effects.",
        ],
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"demand": demand_metrics, "shortage": shortage_metrics,
                      "train_rows": len(train), "test_rows": len(test)}, indent=2))


if __name__ == "__main__":
    main()

