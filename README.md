## Week 11 deliverables (current submission)
- **Live app (revised):** https://ed-supply-dashboard.streamlit.app/ (see the **Predictive Models** page)
- **Evaluation report:** [docs/week11_evaluation_report.md](docs/week11_evaluation_report.md): evaluation design and leakage checks, model vs. baseline comparison, improvements tried and why the final models were kept, user-testing findings and resulting changes, limitations and future work
- **User-testing protocol:** [docs/user_testing_protocol.md](docs/user_testing_protocol.md)
- **Completed user-testing record:** [docs/user_testing_results.md](docs/user_testing_results.md)
- **Evaluation outputs:** `data/modeling/evaluation_predictions.csv`, `data/modeling/error_analysis.csv`, `models/model_metadata.json`
- **Setup and dependencies:** see "Reproduce and run" below and `requirements.txt`
- **Group contributions:** [docs/group_contributions.md](docs/group_contributions.md)
- **Demo video (7 min):** *link to be added*
- **Week 10 submission (preserved for traceability):** tag [`week10-submission`](https://github.com/danmashi91/ed-supply-dashboard/releases/tag/week10-submission)

## Week 10 deliverables
   - **Demo video (7 min):** https://drive.google.com/file/d/1b5Kpco7R8baw41fm1eNZpaUsr5KRJJIW/view?usp=sharing
   - **Live app:** (https://ed-supply-dashboard.streamlit.app/)
   - **Progress report:** docs/week10_progress_report.md
   - **Group contributions:** docs/group_contributions.md

# ED Supply Decision Support — Version 3 (Predictive Validation Dashboard)

Streamlit application using two years (2024-01-01 through 2025-12-31) of **synthetic** emergency-department arrivals and supply data for eight routine consumables. It combines descriptive monitoring with two predictive models, transparent model explanations, and interactive what-if analysis.

Data version: `v1.1-2026-10-19`. Model version: `v3.1-validation-2026-11-02`.

Team: Monica Hall and Abubakar Sani (MS Analytics, Saint Louis University).

## Reproduce and run

Python 3.10 or later is required (on Streamlit Community Cloud, choose Python 3.11 or 3.12). scikit-learn is pinned to 1.7.2 so the saved models load correctly.

```bash
pip install -r requirements.txt
python code/train_predictive_models.py
streamlit run app.py
```

To regenerate the synthetic source and final data before training, first run:

```bash
python code/generate_and_validate.py
```

The app reads the generated CSV and model files without modifying them. The committed model artifacts let the deployed app start without training during launch.

## Pages

1. **Overview** — eight descriptive KPIs, arrival patterns, and a daily supply-risk queue.
2. **Item Explorer** — per-item inventory, demand, receipts, risk, and explanatory status text.
3. **Trends & Mix** — monthly demand, order spend, high-risk days, arrival mix, and an item summary.
4. **Scenario Comparison** — pre-generated low, base, and high demand scenarios.
5. **Predictive Models** — KPI definitions, leakage checks, baseline/original/revised comparisons, error segments, overall importance, individual contributions, constrained what-if inputs, side-by-side comparisons, and an automatic summary.
6. **Data & Assumptions** — item assumptions, validation results, data notes, and filtered-data download.

## Retained KPIs and final validated models

| KPI | Decision use | Algorithm | 2025 holdout result |
|---|---|---|---|
| Predicted item demand during the next seven days | Replenishment planning and avoidance of emergency purchases or excess stock | Arrival-based formula (7-day arrival forecast × units per arrival), retained after Week 11 evaluation; learned models kept as comparisons | WAPE 3.79%; MAE 57.8 units (best learned model 3.84%) |
| Probability of any backorder during the next seven days | Prioritize items for review, ordering, or supplier escalation | Date-safe calibrated, class-weighted logistic regression | ROC-AUC 0.919; average precision 0.252; at the 5.61% screening threshold: 5.06% of item-days flagged, recall 58.33%, precision 9.66%; 24 positive test rows |

## Validation update (Week 11)

**User testing:** one external participant completed two demand what-if scenarios. Raising expected arrivals from 1,285.30 to 1,485.30 increased predicted demand from 540 to 624 units; lowering arrivals to 1,210.30 reduced predicted demand to 508 units. The participant found the comparison understandable and requested more adjustable operational assumptions and clearer recommendations. The application now explains which controls are external, fixed or derived, provides an editable closing-stock control in the shortage scenario, and opens with a **Recommended actions** table. The task record is in `docs/user_testing_results.md` and the evaluation summary is in section 7 of `docs/week11_evaluation_report.md`.

The Week 10 submission is preserved as the git tag `week10-submission` (also branch `week10-submission-snapshot`). Version 3 strengthens the evaluation without using the 2025 final test set for model selection:

- **Shortage (KPI 2):** probability calibration repaired with explicit date-based folds. ROC-AUC 0.919, average precision 0.252, Brier score 0.0075. At the 5.61% screening threshold selected on 2024 data, it flags 5.06% of item-days, with 9.66% precision and 58.33% recall.
- **Demand (KPI 1):** a direction check found that the Week 10 gradient boosting model predicted *lower* demand when 10% more patients were expected on 71% of validation rows. A monotone-constrained model was tried as the improvement; it fixed the direction but was less accurate (6.70% vs. 6.04% validation WAPE). Under the selection rule (must move in the right direction, then lowest 2024 validation error), the arrival-based formula is retained. Learned models remain in the comparison table.
- **What-if:** inputs are labelled as planning inputs, external conditions, fixed observations or derived values; numbers are limited to 2024 ranges (closing stock per item); days of supply follows closing stock; a seven-day forecast inconsistent with the one-day forecast is rejected with no prediction shown; baseline and scenario are compared side by side; the automatic summary covers inputs, comparison, drivers, implication, trade-offs and limitations.

Models are selected on 2024 data only (rolling folds for demand, Oct–Dec 2024 for shortage), refitted on all of 2024, and tested once on 2025. Overall importance uses permutation importance on the holdout set. Individual contributions use one-predictor-at-a-time replacement with a typical training value; they are approximate, may not sum because of interactions, and are not causal effects. Predictor definitions shared by training and the app are in `code/model_features.py`.

## Main files

```text
app.py
requirements.txt
code/generate_and_validate.py
code/train_predictive_models.py
code/model_features.py
data/raw/
data/intermediate/
data/final/
data/modeling/
models/
docs/
```

## Limitations

All observations and outcomes are synthetic, cover only eight supplies, and represent one simulated emergency department. The demand estimate's accuracy reflects known simulation structure; no learned model beat the arrival-based formula, and errors roughly triple during surges. Shortages are rare, so the shortage model's low precision can create many false alerts; it is a screening aid, not an automatic ordering rule. What-if results are model-based scenarios, not guaranteed intervention effects. This prototype is not for real purchasing or clinical decisions.

## Week 10 supporting deliverables

- `docs/week10_progress_report.md` — accomplishments, KPI/model rationale, validation, and limitations.
- `docs/group_contributions.md` — member-contribution summary for both teammates to confirm.
- `docs/video_demonstration_script.md` — timed seven-minute presentation plan.
- `docs/week10_predictive_specification.md` — detailed requirements and rubric traceability.

## Predictive validation deliverables

- `docs/week11_evaluation_report.md` — evaluation design, leakage audit, model comparisons, error analysis, improvements and limitations.
- `docs/user_testing_protocol.md` — structured external-user tasks and observation form.
- `docs/user_testing_results.md` — completed anonymized task record, scenario results, feedback and resulting changes.
- `data/modeling/evaluation_predictions.csv` — saved 2025 predictions and errors.
- `data/modeling/error_analysis.csv` — performance by item, surge status, demand band and supplier event.
- `week10-submission` tag (and `week10-submission-snapshot` branch) — identifiable copy of the Week 10 submission.
