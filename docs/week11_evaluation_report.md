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

WAPE is the primary metric because it expresses total absolute error as a share of total demand and is understandable for aggregate stock planning. MAE reports the average unit error, RMSE emphasizes large misses, and R² is reported only as a supplementary fit measure because item scales differ greatly. The baseline is seven-day forecast arrivals multiplied by the item's assumed units per arrival.

| Version | Validation WAPE | 2025 test WAPE | 2025 MAE |
|---|---:|---:|---:|
| Simple formula baseline | 6.04% | 3.79% | 57.77 units |
| Original Week 10 gradient boosting with item code | 6.61% | 3.85% | 58.67 units |
| Revised gradient boosting without item code | 6.59% | 3.84% | 58.57 units |

The revised model is slightly simpler and marginally better than the original model, but it does not beat the simple baseline. This is expected because the synthetic demand-generation process is primarily driven by arrivals and item use rate. The application displays this result directly. For real deployment, the learned demand model should not replace the formula unless it demonstrates repeatable improvement on real external data.

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

The demand model performs worse during surge periods: 7.08% WAPE and 117.2-unit MAE versus 3.72% WAPE and 56.5-unit MAE on non-surge days. The highest-demand quartile has a mean bias of -84.3 units, indicating underprediction. IV kits and specimen cups have the highest item-level WAPE, at 4.98% and 4.92%. The practical implication is that users should retain additional safety stock and consult the simple arrival-based benchmark during surge or unusually high-demand conditions.

### Shortage errors

Only Gauze, Specimen cups and Underpads contain positive shortage cases in the 2025 test period, with eight cases each. Recall is 75.0% for Gauze, 62.5% for Specimen cups and 37.5% for Underpads. Five other items have no observed test shortages but still produce false alerts. During non-surge periods, recall is only 27.3%, compared with 84.6% during surges. These subgroup estimates are unstable because there are only 24 positive test rows. Users must verify stock, open orders and supplier information before acting.

## 5. Model improvements and retained versions

For demand, the improvement removed item code because units per arrival already represents item scale. This produced nearly identical validation and test performance with a simpler model that is more transferable to a new item. The revised model is retained for the explanatory workflow, while the superior simple benchmark remains visible.

For shortage, the improvement replaced the original raw inventory predictors with scale-free coverage and trend predictors and repaired calibration with explicit calendar-based folds. This increased validation average precision from 0.120 to 0.253 and test average precision from 0.123 to 0.252. The revised model is retained.

## 6. Explanation and what-if improvements

Version 3 adds:

- Plain-language baseline/original/revised performance tables.
- Segment-level error displays and user implications.
- A data-availability and leakage-check explanation.
- Clear separation of overall importance from individual prediction contributions.
- Labels for planning inputs, external scenario conditions, fixed observations and derived fields.
- Development-data ranges on numeric editable inputs.
- Locked historical and calendar fields.
- Automatic recalculation of days of supply when closing stock changes.
- Rejection of an invalid seven-day arrival forecast below the one-day forecast.
- Side-by-side baseline and scenario input tables, predictions and differences.
- A summary that describes inputs, results, trade-offs and non-causal limitations.

## 7. Structured user testing

User testing is pending completion with a person outside the project group. The test protocol is in `docs/user_testing_protocol.md`. This section must be updated with the participant profile, observed tasks, misunderstandings, feedback and at least one resulting application change before final submission. No confidential or real patient data will be shown.

## 8. Remaining limitations and future priorities

- All data and outcomes are synthetic and reflect the generator's assumptions.
- Only eight items and one simulated ED are represented.
- The demand model does not outperform its simple baseline.
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

The identifiable Week 10 copy is preserved in branch `week10-submission-snapshot`. Version 3 work is developed in branch `week11-validation`.
