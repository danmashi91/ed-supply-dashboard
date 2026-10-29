   ## Week 10 deliverables
   - **Demo video (7 min):** https://drive.google.com/file/d/1b5Kpco7R8baw41fm1eNZpaUsr5KRJJIW/view?usp=sharing
   - **Live app:** (https://ed-supply-dashboard.streamlit.app/)
   - **Progress report:** docs/week10_progress_report.md
   - **Group contributions:** docs/group_contributions.md

# ED Supply Decision Support — Version 2 (Week 10 Predictive Dashboard)

Streamlit application using two years (2024-01-01 through 2025-12-31) of **synthetic** emergency-department arrivals and supply data for eight routine consumables. It combines descriptive monitoring with two predictive models, transparent model explanations, and interactive what-if analysis.

Data version: `v1.1-2026-10-19`. Model version: `v2.1-week10-2026-10-25`.

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
5. **Predictive Models** — KPI definitions, holdout performance, overall permutation importance, individual-prediction contributions, editable what-if inputs, baseline comparisons, and an automatically generated scenario summary.
6. **Data & Assumptions** — item assumptions, validation results, data notes, and filtered-data download.

## Week 10 KPIs and models

| KPI | Decision use | Algorithm | 2025 holdout result |
|---|---|---|---|
| Predicted item demand during the next seven days | Replenishment planning and avoidance of emergency purchases or excess stock | Gradient boosting regressor | WAPE 3.84% (no-model benchmark 3.79%); MAE 58.6 units; RMSE 95.2; R² 0.995 |
| Probability of any backorder during the next seven days | Prioritize items for review, ordering, or supplier escalation | Calibrated, class-weighted logistic regression | ROC-AUC 0.915; average precision 0.224; at the 7.3% screening threshold: 8.8% of item-days flagged, recall 75%, precision 7.1%; 24 positive test rows |

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

All observations and outcomes are synthetic, cover only eight supplies, and represent one simulated emergency department. The demand model's strong accuracy reflects known simulation structure, and it does not beat a simple arrival-based formula. Shortages are rare, so the shortage model's low precision can create many false alerts; it is a screening aid, not an automatic ordering rule. What-if results are model-based scenarios, not guaranteed intervention effects. This prototype is not for real purchasing or clinical decisions.

## Week 10 supporting deliverables

- `docs/week10_progress_report.md` — accomplishments, KPI/model rationale, validation, and limitations.
- `docs/group_contributions.md` — member-contribution summary for both teammates to confirm.
- `docs/video_demonstration_script.md` — timed seven-minute presentation plan.
- `docs/week10_predictive_specification.md` — detailed requirements and rubric traceability.
