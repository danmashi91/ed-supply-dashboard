ED SUPPLY DECISION SUPPORT — WEEK 10
====================================

Team: Monica Hall and Abubakar Sani
Data: synthetic ED arrivals and inventory, 2024-01-01 to 2025-12-31

Reproduce and run
-----------------
1. pip install -r requirements.txt
2. python code/generate_and_validate.py   (optional: recreates source data)
3. python code/train_predictive_models.py
4. streamlit run app.py

Predictive extension
--------------------
- KPI 1: predicted item demand during the next seven days, modeled with a
  gradient boosting regressor and compared with a no-model benchmark
  (arrival forecast x units per arrival).
- KPI 2: probability of any backorder during the next seven days, modeled
  with class-weighted logistic regression plus probability calibration.
- Models are chosen on 2024 data only and tested once on 2025.
- Overall importance: permutation importance on the chronological 2025
  holdout set.
- Individual contributions: one-feature-at-a-time reference replacement;
  predictive, approximate, and explicitly non-causal.
- What-if inputs compare the updated result with a clearly labeled baseline
  and automatically generate a scenario summary.

Important limitation
--------------------
The data and outcomes are synthetic. The shortage model has low precision
because positive cases are rare, and the demand model does not beat the
simple arrival-based benchmark on this synthetic data. Use all predictions for demonstration and
decision screening only, not automatic purchasing or clinical decisions.

See README.md and docs/week10_progress_report.md for complete details.
