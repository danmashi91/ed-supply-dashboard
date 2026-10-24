# Week 10 Progress Report

## Project status

The group has expanded the ED Supply Decision Support application from a descriptive Streamlit dashboard into a predictive decision-support prototype. The original decision-maker remains the emergency-department inventory or supply manager, and the original problem remains maintaining supply availability under fluctuating demand and supplier disruption while controlling inventory and purchasing costs.

## Work completed

- Preserved the existing two-year synthetic project scope, data definitions, filters, risk queue, item exploration, trends, and scenario comparison.
- Defined two forward-looking KPIs: predicted seven-day item demand and probability of a backorder during the next seven days.
- Created reproducible seven-day target variables from item-day records.
- Used a chronological split: 2024 for training and 2025 for testing.
- Trained a gradient boosting demand model and class-weighted logistic shortage model.
- Saved fitted models, model metadata, modeling data, and permutation-importance results.
- Added a Predictive Models page with KPI explanations, test metrics, global importance, editable what-if inputs, baseline comparisons, local contributions, and automatic summaries.
- Added non-causation language and limitations in the interface and documentation.
- Updated setup instructions and prepared a timed video demonstration script.

## Current model results

The seven-day demand model achieved 3.85% WAPE, 58.671-unit MAE, 95.369-unit RMSE, and R² of 0.9950 on the chronological 2025 holdout. These results show that the model reproduces the patterns in the synthetic generator well.

The seven-day shortage model achieved ROC-AUC 0.8712 and recall 0.7083, but precision was only 0.0425 and F1 was 0.0802. There were only 24 positive cases among 2,864 test observations. The shortage output is therefore presented as a screening estimate that requires manager review, not an automatic order trigger.

## Validation completed

- Both target windows use future days only and remove rows with incomplete seven-day outcomes.
- Predictor and target columns were checked for missing values before fitting.
- Model pipelines handle numeric scaling and unknown categorical levels consistently.
- Training and testing preserve chronology.
- Saved artifacts were regenerated successfully from the training script.
- The original data pipeline continues to report 13 of 13 validation checks passing.
- The application exposes the exact holdout measures stored by the training pipeline.
- Syntax and launch tests are documented in the repository review before submission.

## Remaining submission actions

1. Both group members should confirm the contribution wording in `docs/group_contributions.md`.
2. Record the seven-minute demonstration using `docs/video_demonstration_script.md`, with both members visibly or audibly participating if required by the instructor.
3. Upload the video to the repository release, OneDrive folder, or another instructor-accessible location and place the link in the README.
4. Push the Week 10 branch, open or merge a pull request, and verify the deployed Streamlit application.
5. Submit one accessible repository/folder link in Canvas for the group.

## Risks and limitations

The application remains a classroom prototype based entirely on synthetic data. Results are appropriate for demonstrating the proposed workflow but cannot establish real-world accuracy, clinical safety, or financial savings. The shortage outcome is particularly sparse. Future work should validate with authorized operational data, use multiple sites and more items, evaluate probability calibration and alert thresholds, and conduct user testing with supply managers.

