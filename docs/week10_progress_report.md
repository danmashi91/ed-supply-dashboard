# Week 10 Progress Report

## Project status

The group has expanded the ED Supply Decision Support application from a descriptive Streamlit dashboard into a predictive decision-support prototype. The original decision-maker remains the emergency-department inventory or supply manager, and the original problem remains maintaining supply availability under fluctuating demand and supplier disruption while controlling inventory and purchasing costs.

## Work completed

- Preserved the existing two-year synthetic project scope, data definitions, filters, risk queue, item exploration, trends, and scenario comparison.
- Defined two forward-looking KPIs: predicted seven-day item demand and probability of a backorder during the next seven days.
- Created reproducible seven-day target variables from item-day records.
- Used a chronological split: 2024 for training and 2025 for testing.
- Trained a gradient boosting demand model and a calibrated, class-weighted logistic shortage model.
- Reviewed the first model versions and revised them: removed duplicate and collinear predictors, added probability calibration, compared candidates on 2024 validation data only, and added a no-model benchmark for demand.
- Saved fitted models, model metadata, modeling data, and permutation-importance results.
- Added a Predictive Models page with KPI explanations, test metrics, global importance, editable what-if inputs, baseline comparisons, local contributions, and automatic summaries.
- Added non-causation language and limitations in the interface and documentation.
- Updated setup instructions and prepared a timed video demonstration script.

## Current model results

The seven-day demand model (gradient boosting) achieved 3.84% WAPE, 58.6-unit MAE, 95.2-unit RMSE, and R² of 0.995 on the chronological 2025 holdout. A no-model benchmark (seven-day arrival forecast × units used per arrival) scored 3.79%. The model therefore matches but does not beat the formula, which is expected because the synthetic generator creates demand from that formula. Overall importance shows that units per arrival and recent demand drive predictions; the arrival forecast adds little beyond recent demand, which already reflects current arrival levels.

The seven-day shortage model (calibrated logistic regression) achieved ROC-AUC 0.915 and average precision 0.224 (a random ranking would score about 0.008). At a screening threshold of 7.3%, chosen on 2024 validation data, it flags 8.8% of item-days and catches 75% of shortages, but precision is 7.1% because there were only 24 positive cases among 2,864 test observations. Compared with the first version, it flags fewer item-days (8.8% vs. 14%) while catching slightly more shortages (75% vs. 71%), and its average predicted probability fell from 26% to 2.9%. Low stock against recent demand, a rising demand trend, surges, and supplier delays raise the estimate. The output is a screening estimate that requires manager review, not an automatic order trigger.

## Validation completed

- Both target windows use future days only and remove rows with incomplete seven-day outcomes.
- Predictor and target columns were checked for missing values before fitting.
- Model pipelines handle numeric scaling and unknown categorical levels consistently.
- Training and testing preserve chronology.
- Saved artifacts were regenerated successfully from the training script.
- The original data pipeline continues to report 13 of 13 validation checks passing.
- The application exposes the exact holdout measures stored by the training pipeline.
- Every page, both what-if forms, empty-filter cases, and the original descriptive acceptance values were tested with Streamlit's app-testing tool after the model revision.
- scikit-learn is pinned (1.7.2) so the saved models load the same way locally and on Streamlit Community Cloud.

## Remaining submission actions

1. Both group members should confirm the contribution wording in `docs/group_contributions.md`.
2. Record the seven-minute demonstration using `docs/video_demonstration_script.md`, with both members visibly or audibly participating if required by the instructor.
3. Upload the video to the repository release, OneDrive folder, or another instructor-accessible location and place the link in the README.
4. Push the Week 10 branch, open or merge a pull request, and verify the deployed Streamlit application.
5. Submit one accessible repository/folder link in Canvas for the group.

## Risks and limitations

The application remains a classroom prototype based entirely on synthetic data. Results are appropriate for demonstrating the proposed workflow but cannot establish real-world accuracy, clinical safety, or financial savings. The shortage outcome is particularly sparse. Future work should validate with authorized operational data, use multiple sites and more items, evaluate probability calibration and alert thresholds, and conduct user testing with supply managers.

