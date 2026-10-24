# ED Supply Decision Support — Version 2 (Week 10 Predictive Dashboard)

Streamlit application using two years (2024-01-01 through 2025-12-31) of **synthetic** emergency-department arrivals and supply data for eight routine consumables. It combines descriptive monitoring with two predictive models, transparent model explanations, and interactive what-if analysis.

Data version: `v1.1-2026-10-19`. Model version: `v2.0-week10-2026-10-24`.

Team: Monica Hall and Abubakar Sani (MS Analytics, Saint Louis University).

## Reproduce and run

Python 3.10 or later is required.

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
| Predicted item demand during the next seven days | Replenishment planning and avoidance of emergency purchases or excess stock | Gradient boosting regressor | WAPE 3.85%; MAE 58.7 units; RMSE 95.4; R² 0.995 |
| Probability of any backorder during the next seven days | Prioritize items for review, ordering, or supplier escalation | Class-weighted logistic regression | ROC-AUC 0.871; recall 70.8%; precision 4.25%; F1 0.080; 24 positive test rows |

Training uses 2024 rows and testing uses 2025 rows, preserving time order. Overall importance uses permutation importance on the holdout set. Individual contributions use one-feature-at-a-time replacement with a training reference value; they are approximate, may not sum because of interactions, and are not causal effects.

## Main files

```text
app.py
requirements.txt
code/generate_and_validate.py
code/train_predictive_models.py
data/raw/
data/intermediate/
data/final/
data/modeling/
models/
docs/
```

## Limitations

All observations and outcomes are synthetic, cover only eight supplies, and represent one simulated emergency department. The demand model's strong accuracy partly reflects known simulation structure. Shortages are rare, so the shortage model's low precision can create many false alerts; it is a screening aid, not an automatic ordering rule. What-if results are model-based scenarios, not guaranteed intervention effects. This prototype is not for real purchasing or clinical decisions.

## Week 10 supporting deliverables

- `docs/week10_progress_report.md` — accomplishments, KPI/model rationale, validation, and limitations.
- `docs/group_contributions.md` — member-contribution summary for both teammates to confirm.
- `docs/video_demonstration_script.md` — timed seven-minute presentation plan.
- `docs/week10_predictive_specification.md` — detailed requirements and rubric traceability.
