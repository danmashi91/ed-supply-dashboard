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
- **Predictors and relevance:**
  - *Units used per arrival* sets each item's scale (a patient uses far more gloves than specimen cups).
  - *Previous-day and trailing seven-day item demand* are the most direct guide to the item's current level and trend.
  - *Current, one-day and seven-day arrival forecasts*: more patients means more supplies used.
  - *Surge indicator*: surges keep demand above normal for several days.
  - *Day of week, weekend, holiday, month*: calendar patterns in arrivals and use.
  - The item code was removed: it duplicated units-used-per-arrival and split importance between two predictors that mean the same thing.
- **Algorithm:** `GradientBoostingRegressor` with Huber loss.
- **Rationale:** demand depends on combinations of item scale, recent use, calendar and surges that do not combine linearly. Gradient boosting captures these patterns without a fixed formula, and Huber loss limits the influence of extreme surge days.
- **Benchmark:** a no-model formula, seven-day arrival forecast × units used per arrival, is reported next to the model. Any model must beat this benchmark before it is trusted on real data.

### 3.2 Shortage model

- **Target:** `target_shortage_next_7d`, equal to 1 if any backorder occurs in the next seven days and 0 otherwise.
- **Predictors and relevance** (raw values the user sees, and the scale-free model predictors derived from them in `code/model_features.py`):
  - *Days of supply* and *supplier lead time* → coverage gap = days of supply − lead time: whether stock lasts until a new order could arrive.
  - *Closing stock* and *trailing seven-day demand* → stock cover ratio = closing stock ÷ recent weekly demand: days of physical cover.
  - *Previous-day demand* ÷ *trailing seven-day average* → demand trend: a rising trend uses stock faster.
  - *Seven-day arrival forecast* ÷ *previous 28-day average arrivals* → arrival ratio: above-normal patient pressure.
  - *Surge indicator* and *supplier event* (late or short deliveries).
  - Ratios make the predictors comparable across items of very different size. The first version's item code, item-level constants (use rate, service-level target), and the overlapping inventory position / open-order / backorder columns were removed: they were perfectly collinear with each other and produced misleading coefficients (for example, more open orders appearing to *raise* shortage risk).
- **Algorithm:** class-weighted logistic regression with sigmoid probability calibration (`CalibratedClassifierCV`, four time-ordered folds).
- **Rationale:** the outcome is binary and rare. Logistic regression outputs a probability and is easy to explain; class weighting helps it learn from scarce shortage examples; calibration then rescales the weighted scores so a stated probability is close to a real frequency. Without calibration, the first version's average output was 26% against an actual shortage rate under 1%.

## 4. Training, selection and validation design

- Source observations are synthetic item-day records.
- Forward targets exclude the final seven calendar days because their outcome windows are incomplete.
- **Model selection uses 2024 only.** Demand candidates are compared on four rolling validation folds (May–Jun, Jul–Aug, Sep–Oct, Nov–Dec 2024, each trained on all earlier dates). Shortage candidates are fitted on Jan–Sep 2024 and compared on Oct–Dec 2024 using average precision, which suits rare outcomes.
- The shortage screening threshold is the value that maximized F1 on Oct–Dec 2024. It was not tuned on the test set.
- The chosen models are refitted on all of 2024 and scored once on 2025-01-01 to 2025-12-24.
- Training rows: 2,928; test rows: 2,864. Fixed random state: 20261024. scikit-learn 1.7.2 (pinned).

| Demand candidate | Mean validation WAPE (2024 rolling folds) |
|---|---|
| Benchmark: arrival forecast × units per arrival (no model) | 6.04% |
| **Gradient boosting without item code (chosen)** | 6.61% |
| Gradient boosting with item code (first Week 10 version) | 6.59% |
| Gradient boosting with arrival-driven expected demand | 6.70% |
| Linear regression on expected and recent demand | 6.94% |

Fitted candidates within 0.25 percentage points are treated as tied; the simplest design (no item code) is preferred.

| Shortage candidate | Validation ROC-AUC | Validation average precision |
|---|---|---|
| **Logistic regression on coverage and trend predictors (chosen)** | 0.811 | 0.204 |
| Logistic regression on original predictors (first Week 10 version) | 0.707 | 0.082 |

**2025 test results**

| Model | Measures | Result |
|---|---|---|
| Demand | WAPE; benchmark WAPE; MAE; RMSE; R² | 3.84%; 3.79%; 58.6 units; 95.2 units; 0.995 |
| Shortage | ROC-AUC; average precision; screening threshold; item-days flagged; recall; precision; F1 | 0.915; 0.224; 7.3%; 8.8%; 75.0%; 7.1%; 0.130 |
| Shortage calibration | Observed rate; mean predicted probability; Brier score (base-rate Brier) | 0.84%; 2.9%; 0.0090 (0.0087) |

**Interpretation.** The demand model is accurate but does not beat the simple arrival-based benchmark. That is expected, because the synthetic generator creates demand from almost exactly that formula; on real data the benchmark is the bar to beat. R² is high mainly because items differ greatly in scale. The shortage model ranks risk well (a random ranking would score average precision ≈ 0.008), but only 24 test rows are positive, so most alerts are false alarms. Its probabilities are about three times the 2025 shortage rate because shortages were more common in the 2024 training year; probabilities must be recalibrated on real data before operational use.

## 5. Explainability requirements

**Overall importance** uses test-set permutation importance on the raw, user-facing predictors. Each predictor is shuffled across the 2025 test data (8 repeats), derived predictors are recomputed, and the loss of accuracy is shown with ± one standard deviation (demand: increase in mean absolute error in units; shortage: drop in ROC-AUC). It describes what the fitted model relies on across the test set.

**Individual contributions** use one-predictor-at-a-time reference replacement. For the selected record, one raw predictor is set to its typical 2024 value (median, or most common for supplier event) and the prediction is recomputed; the difference is that predictor's approximate contribution. They are shown for the baseline prediction and again for every what-if scenario. Interactions mean contributions may not sum exactly.

**Contextual interpretation** is shown beside both charts: direction (pushes the prediction up or down), magnitude (units or percentage points), the predictor's value compared with its typical value, and practical meaning for the inventory manager.

Neither technique demonstrates causation. The interface states this next to the results.

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
| Individual contributions | Baseline and per-scenario local contribution charts and text |
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
- Shortage events are rare, so precision at the 7.3% screening threshold is low (about 1 alert in 14 is followed by a real shortage).
- The demand model does not beat a simple arrival-based formula on this synthetic data.
- Probability calibration must be reassessed with real outcomes before operational use.
- User-provided combinations may be outside the training distribution; the prototype does not enforce multivariate plausibility.
- Recommendations require inventory-manager review and must not be treated as clinical advice or an automatic purchase order.

