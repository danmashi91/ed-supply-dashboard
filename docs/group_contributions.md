# Group Member Contributions

## Weeks 12-13: application validation and evidence

### Monica Hall

- Translated the Weeks 12-13 instructions into project-specific requirements, acceptance criteria and test rows.
- Completed the validation matrix, including supported claims, partial passes, the untested UI reconciliation, limitations and corrective actions.
- Prepared the two-to-three-page validation memo and the evidence index.
- Ran the data generator, model training and automated application-load test in a new Python environment and documented the reproduced results.
- Prepared the typical and challenging end-to-end decision walkthroughs.
- Organized the Week 12 submission checklist and current evidence package.

### Abubakar Sani

- Built and deployed the core Streamlit application used in the validation.
- Developed the dashboard pages, navigation, filters, charts and recommended-actions display.
- Contributed the predictive-model implementation, scenario controls, input checks and automatic summaries evaluated in the matrix.
- Arranged Tester A's external review and implemented the recommended-actions and layout improvements arising from user feedback.
- Reviewed and merged the Week 11 application and documentation that form the validated version 3.1 baseline.

### Shared responsibilities

- Review the completed validation package and confirm that claims match the current application.
- Record the seven-minute demonstration and twelve-minute matrix walkthrough.
- Add accessible video links, confirm repository access and submit the final GitHub link.

## Week 11: model evaluation, improvement and user testing

### Monica Hall

- Translated the Week 11 rubric into evaluation, application and deliverable requirements.
- Preserved the Week 10 submission in the `week10-submission-snapshot` branch.
- Found and fixed a temporal-leakage risk in shortage-model calibration by replacing row-order folds with explicit date-based folds.
- Built the baseline/original/revised comparisons for both KPIs and saved the 2025 evaluation predictions.
- Added the segment error analysis (by item, surge status, demand band and supplier event) and the leakage-check documentation.
- Added predictor-control labels (planning, external, fixed, derived), development-data input ranges, locked historical fields, days-of-supply recalculation and side-by-side scenario inputs.
- Drafted the Week 11 evaluation report and the structured user-testing protocol.
- Created and managed the Week 11 branch and pull request.

### Abubakar Sani

- Reviewed the Week 11 pull request against the rubric and reproduced all results.
- Identified that the Week 10 demand models predicted lower demand when more patients were expected (71–73% of validation rows); added a direction check to model selection.
- Implemented and evaluated a monotone-constrained gradient boosting model as the demand improvement; retained the arrival-based formula under the selection rule and documented the result.
- Tightened what-if checks: per-item stock ranges and a seven-day vs. one-day arrival-forecast consistency check that rejects implausible scenarios.
- Added plain-language performance statements, readable comparison and error tables, and trade-offs to the automatic scenario summary.
- Arranged the external user test and implemented the resulting changes: the Recommended actions table and a less crowded layout.
- Updated the evaluation report (comparisons, error analysis, improvements, user testing, limitations) and README; created the `week10-submission` tag.
- Prepared the in-class presentation deck and demo script.

### Shared responsibilities

- Agreed the retained models and their interpretation and limitations.
- Reviewed and merged the Week 11 changes and checked the deployed application.
- Record the seven-minute demonstration video, submit the repository link, and present in class on November 5.

## Week 10: predictive models and what-if analysis

### Abubakar Sani

- Developed and deployed the initial Streamlit dashboard.
- Implemented the descriptive pages, navigation, visualizations, filtering, and deployment structure.
- Helped define the ED supply-management problem, simulated workflow, and initial KPI presentation.
- Reviewed and tested the Week 10 predictive extension.
- Improved the shortage model through probability calibration and revised predictors.
- Added the demand benchmark, model-selection comparisons, and clearer model explanations.
- Updated the application and supporting documentation after model review.

### Monica Hall

- Reviewed the Week 10 rubric and translated it into predictive, application, and presentation requirements.
- Developed and coordinated the first predictive extension for seven-day item demand and seven-day shortage probability.
- Added model-validation reporting, overall predictor importance, individual prediction contributions, and contextual interpretation.
- Added baseline-versus-updated what-if analysis and automatically generated scenario summaries.
- Updated the setup instructions, predictive specification, progress report, limitations, and presentation documentation.
- Created and managed the Week 10 GitHub branch and pull request.
- Verified the merged application and Predictive Models page on Streamlit.

### Shared responsibilities

- Reviewed the final KPI definitions, model interpretation, and limitations.
- Tested the final deployed application and repository access.
- Prepared the seven-minute demonstration plan.
- Will record and edit the demonstration video and submit the final repository link.
