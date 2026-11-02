# Predictive Model Evaluation and Application Validation Report

## 1. Purpose and continuity

This report evaluates the Week 10 ED Supply Decision Support application for one simulated emergency department. The intended decision-maker remains the ED inventory manager. The two retained KPIs are (1) units requested during the next seven days and (2) the probability of any backorder during the next seven days. No project-scope or KPI change was made.

All data are synthetic. Results therefore assess internal prototype behavior and fitness for a classroom decision-support demonstration, not readiness for hospital purchasing or clinical use.

## 2. Evaluation design

Each modeling row represents one item at the end of day *t*. Predictors must be available by that point; targets cover days *t+1* through *t+7*. The chronological design is:

- Development period: January 1–December 31, 2024 (2,928 item-days).
- Demand selection: four rolling-origin validation windows within 2024.
- Shortage fit/validation split: January–September 2024 for fitting and October–December 2024 for validation.
- Final test: January 1–December 24, 2025 (2,864 item-days). The final seven calendar days are omitted because a complete seven-day outcome is unavailable.

Candidate models, predictor choices and the shortage screening threshold were selected using 2024 only. The 2025 test period was used once for final reporting and not for repeated tuning.

### Data-leakage checks

1. Targets are calculated only from days *t+1* through *t+7*.
2. Previous-day demand, seven-day trailing demand and 28-day average arrivals use shifted historical values.
3. Arrival forecasts are generated only from observations available through day *t*.
4. Closing stock, days of supply and supplier status are values available at the end-of-day inventory review.
5. Model and threshold selection use 2024 only.
6. The Week 10 shortage calibration originally used `TimeSeriesSplit` on rows stored by item and date. That could mix calendar periods across items. Version 3 replaces it with explicit expanding folds defined by calendar date, so every calibration training date precedes its validation dates.

## 3. Metrics and baselines

### Seven-day demand

WAPE is the primary metric because it expresses total absolute error as a share of total demand and is understandable for aggregate stock planning. MAE reports the average unit error, RMSE emphasizes large misses, and R² is reported only as a supplementary fit measure because item scales differ greatly. The baseline is the seven-day arrival forecast multiplied by the item's assumed units per arrival.

Because the application is used for what-if analysis, a second requirement was added during development: **the prediction must not fall when more patients are expected.** For every candidate, all arrival inputs were raised by 10% on each 2024 validation row, and the share of rows where the prediction went down was recorded.

| Version | Validation WAPE (2024 folds) | Rows where +10% patients lowers demand | 2025 test WAPE | 2025 MAE | Retained |
|---|---:|---:|---:|---:|:---:|
| Simple formula baseline | 6.04% | 0% | 3.79% | 57.8 units | ✅ |
| Original Week 10 gradient boosting (with item code) | 6.61% | 73% | 3.85% | 58.7 units | |
| Week 10 revision (without item code) | 6.59% | 71% | 3.84% | 58.6 units | |
| Week 11 improvement: monotone-constrained gradient boosting | 6.70% | 0% | 4.65% | 70.9 units | |

The learned models do not provide a meaningful improvement over the baseline. The Week 10 models also respond in the wrong direction to patient volume on most rows: their what-if results told a user that more patients meant *less* demand. Under the selection rule (respond in the right direction, then lowest 2024 validation WAPE), the **arrival-based formula is retained** as the KPI 1 prediction. This is expected for synthetic data generated mainly from arrivals × use rate; with real data containing other demand drivers (acuity mix, seasonal outbreaks, practice changes), a learned model may add value and should be re-tested against the same baseline and direction check.

### Seven-day shortage probability

Average precision is the primary ranking metric because shortages are rare. ROC-AUC measures general ranking, Brier score measures probability accuracy, and precision/recall describe the practical alert trade-off. The baseline assigns the 2024 shortage prevalence to every row and cannot rank items.

| Version | Validation AP | Test ROC-AUC | Test AP | Brier | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Constant-rate baseline | 3.80% | 0.500 | 0.008 | 0.0087 | 0.0% | 0.0% |
| Original raw-predictor logistic model | 11.98% | 0.920 | 0.123 | 0.0082 | 9.38% | 62.50% |
| Revised coverage/trend model with date-safe calibration | 25.33% | 0.919 | 0.252 | 0.0075 | 9.66% | 58.33% |

The revised shortage model is retained. Its test average precision is approximately double that of the original model and its Brier score is better. At the 5.61% threshold selected on 2024 validation data, it flags 5.06% of test item-days and catches 58.33% of shortages. Precision remains low, so it is suitable only for prioritizing a review queue.

## 4. Error analysis

### Demand errors

Errors are concentrated in surge periods: 11.1% WAPE and 184-unit MAE during surges versus 3.5% and 53 units otherwise. During surges the estimate is too low by about 167 units per item-week on average, because the arrival forecast is built from past arrivals and does not anticipate a surge. Specimen cups (4.95%) and underpads (4.48%) have the highest item-level WAPE. The practical implication is that during a surge the manager should hold additional safety stock and use the what-if to test a higher arrival forecast rather than rely on the baseline estimate.

### Shortage errors

Only Gauze, Specimen cups and Underpads contain positive shortage cases in the 2025 test period, with eight cases each. Recall is 75.0% for Gauze, 62.5% for Specimen cups and 37.5% for Underpads. Five other items have no observed test shortages but still produce false alerts. During non-surge periods, recall is only 27.3%, compared with 84.6% during surges. These subgroup estimates are unstable because there are only 24 positive test rows. Users must verify stock, open orders and supplier information before acting.

## 5. Model improvements and retained versions

**Demand.** Two improvements were evaluated with the same rolling-fold procedure on 2024 data. (1) Removing the item code (Week 10 revision) gave the same accuracy with a simpler model, but the direction check showed it still lowered predictions for more patients on 71% of rows. (2) A monotone-constrained gradient boosting model, which is not allowed to lower its prediction when arrivals, recent demand or use rate rise, fixed the direction problem but was less accurate (6.70% validation WAPE). Neither improvement beat the formula baseline, so the formula is retained and the learned models are documented as unsuccessful but informative improvements. The best learned model is saved for reference (`models/demand_learned_model.joblib`).

**Shortage.** The improvement replaced the original raw inventory predictors with scale-free coverage and trend predictors and repaired calibration with explicit calendar-based folds. This increased validation average precision from 0.120 to 0.253 and test average precision from 0.123 to 0.252, with a better Brier score. The revised model is retained as a screening tool.

## 6. Explanation and what-if improvements

Version 3 adds:

- Plain-language performance statements beside the technical metrics (for example, "roughly 1 in 10 flagged items actually runs short; it catches about 6 in 10 shortages").
- Baseline/original/revised comparison tables, including the direction check for demand.
- Segment-level error displays whose wording is computed from the saved error analysis, with user implications.
- A data-availability and leakage-check explanation.
- Clear separation of overall importance from individual prediction contributions.
- Labels for planning inputs, external conditions, fixed observations and derived fields.
- Plausible ranges from the 2024 development data, with closing stock limited to each item's own range.
- Locked historical, calendar and item-characteristic fields.
- Automatic recalculation of days of supply when closing stock changes.
- A cross-check that the seven-day arrival forecast stays within 0.75–1.40× seven times the one-day forecast (2024 values were 0.83–1.25×). Inconsistent scenarios are rejected with an explanation and no prediction is shown.
- A side-by-side table of baseline and scenario inputs, predictions, differences and (for shortage) whether the item would be flagged.
- An automatic summary covering inputs, comparison, drivers, implication, trade-offs and limitations, stated as model-based scenarios rather than causal effects.

## 7. Structured user testing

User testing is pending completion with a person outside the project group. The test protocol is in `docs/user_testing_protocol.md`. This section must be updated with the participant profile, observed tasks, misunderstandings, feedback and at least one resulting application change before final submission. No confidential or real patient data will be shown.

## 8. Remaining limitations and future priorities

- All data and outcomes are synthetic and reflect the generator's assumptions.
- Only eight items and one simulated ED are represented.
- No learned demand model outperformed the simple arrival-based formula; demand errors roughly triple during surges, when the estimate is too low.
- Only 24 shortage-positive test rows exist, concentrated in three items.
- Subgroup performance cannot be estimated reliably where no positive cases occur.
- The selected threshold represents one precision/recall trade-off and should be chosen with a real manager's error costs.
- Local contributions are approximate model sensitivities and may not add exactly because predictors interact.
- Scenario results are predictive associations, not causal intervention effects.
- Real data, external-site validation, monitoring for drift, probability recalibration and decision-cost evaluation are required before operational use.

## 9. Reproducibility

Run the following commands from the repository root:

```bash
pip install -r requirements.txt
python code/generate_and_validate.py
python code/train_predictive_models.py
streamlit run app.py
```

The identifiable Week 10 copy is preserved as tag `week10-submission` (and branch `week10-submission-snapshot`). Version 3 work is developed in branch `week11-validation`.
