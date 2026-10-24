# Week 10 Predictive Application Specification

## 1. Continuity and decision-maker

**Project:** ED Supply Decision Support.  
**Decision-maker:** Emergency-department inventory or supply manager.  
**Problem:** Routine consumables must be available despite changing patient volume and supplier disruption, while avoiding unnecessary inventory and emergency purchasing.  
**Supported decisions:** which item to review first; how much near-term demand to expect; whether shortage risk warrants ordering or supplier escalation; and how a model prediction changes under a hypothetical scenario.

The decision-maker, problem, synthetic data scope, and descriptive dashboard remain continuous with earlier work. Week 10 adds predictive models and model explanations; it does not replace the existing descriptive KPIs or the pre-generated scenario page.

## 2. KPI requirements

| KPI | Definition | Decision relevance | Organizational link |
|---|---|---|---|
| Seven-day item demand | Predicted sum of units requested from day t+1 through t+7 for one item | Supports replenishment quantity and timing review | Availability, lower emergency purchasing, and less excess stock |
| Seven-day shortage probability | Estimated probability that any backorder occurs from day t+1 through t+7 for one item | Supports review-queue prioritization and supplier escalation | Service continuity and operational resilience |

These are forward-looking decision KPIs. Existing fill rate, order spend, backorder days, forecast WAPE, days of supply, and the descriptive risk proxy remain monitoring metrics rather than additional Week 10 predictive targets.

## 3. Model design

### 3.1 Demand model

- **Target:** `target_demand_next_7d`.
- **Predictors:** current and forecast ED arrivals; prior-day item demand; trailing seven-day item demand; use rate per arrival; calendar fields; surge, weekend, and holiday indicators; and item identity.
- **Algorithm:** `GradientBoostingRegressor` with Huber loss.
- **Rationale:** demand contains nonlinear relationships and interactions among patient volume, item identity, recent use, calendar patterns, and surges. Gradient boosting captures those patterns without requiring a linear form, and Huber loss reduces sensitivity to extreme simulated demand.

### 3.2 Shortage model

- **Target:** `target_shortage_next_7d`, equal to 1 if any future backorder occurs in the next seven days and 0 otherwise.
- **Predictors:** seven-day arrivals forecast, recent demand, inventory position, closing stock, open orders, current backorders, days of supply, lead time, use rate, service-level target, surge indicator, item identity, and current supplier event.
- **Algorithm:** class-weighted logistic regression.
- **Rationale:** the outcome is binary and rare. Logistic regression returns an understandable probability; class weighting increases attention to scarce positive examples; and the simpler structure supports transparent review. It is used for screening, not automatic action.

## 4. Training and validation design

- Source observations are synthetic item-day records.
- Forward targets exclude the final seven calendar days because their outcome windows are incomplete.
- Training period: 2024-01-01 through 2024-12-31.
- Test period: 2025-01-01 through 2025-12-24.
- Split method: chronological, avoiding random leakage from future dates into the training period.
- Training rows: 2,928; test rows: 2,864.
- Fixed model random state: 20261024.

| Model | Holdout measures | Result |
|---|---|---|
| Demand | WAPE; MAE; RMSE; R² | 3.85%; 58.671 units; 95.369 units; 0.9950 |
| Shortage | ROC-AUC; recall; precision; F1; Brier score | 0.8712; 0.7083; 0.0425; 0.0802; 0.1170 |

The demand result shows strong predictive fit on the simulation. It should not be generalized to real hospital data. The shortage model separates higher-risk from lower-risk cases reasonably well, but only 24 test rows are positive and precision is low. Most threshold-based alerts may therefore be false positives.

## 5. Explainability requirements

**Overall importance** uses test-set permutation importance. Each predictor is shuffled while other inputs remain unchanged, and the resulting decrease in holdout performance is displayed. It describes reliance of the fitted model across the test set.

**Individual contributions** use one-feature-at-a-time reference replacement. For a selected record, one predictor is replaced with its training median or mode; the difference from the full prediction is shown as that predictor's approximate contribution. This describes local model sensitivity. Interactions mean contributions may not sum exactly.

Neither technique demonstrates causation. The interface must state this next to the results.

## 6. What-if and summary requirements

For either model, a user selects a baseline item and date. The page must:

1. Display the stored baseline prediction.
2. Provide editable, operationally meaningful predictor inputs.
3. Run the fitted model on the updated values.
4. Display baseline, updated prediction, and difference.
5. Plot approximate contributions to the updated prediction.
6. Automatically summarize changed inputs, prediction and change, practical implication, and limitations.

All outputs are labeled model-based scenarios. The interface must not promise that changing a real-world input will cause the modeled change.

## 7. Acceptance criteria and rubric traceability

| Rubric requirement | Acceptance evidence |
|---|---|
| Two or three KPIs identified and justified | Two KPI definitions and decision rationale appear in this specification, progress report, README, and app |
| Predictors and algorithms explained | Sections 3.1 and 3.2; executable training script |
| Overall predictor importance | Permutation-importance chart for each model |
| Individual contributions | Per-scenario local contribution chart and text |
| Contextual interpretation | Direction, magnitude, units, practical implication, and non-causation warnings |
| What-if analysis | Editable inputs with labeled baseline, updated value, and change |
| Automatic summary | Generated after every submitted what-if scenario |
| Data and code | `data/`, `models/`, `code/`, and `app.py` |
| Setup instructions | `README.md`, `README.txt`, and `requirements.txt` |
| Progress report | `docs/week10_progress_report.md` |
| Seven-minute video | Recording guide in `docs/video_demonstration_script.md`; final video must still be recorded and uploaded |
| Member contributions | `docs/group_contributions.md`, to be confirmed by both members before submission |

## 8. Known limitations and safeguards

- Synthetic patterns can make performance look better than performance on operational data.
- Eight consumables and one simulated department do not represent the diversity of a health system.
- Shortage events are rare, producing low precision at the displayed 0.50 screening threshold.
- Probability calibration must be reassessed with real outcomes before operational use.
- User-provided combinations may be outside the training distribution; the prototype does not enforce multivariate plausibility.
- Recommendations require inventory-manager review and must not be treated as clinical advice or an automatic purchase order.

