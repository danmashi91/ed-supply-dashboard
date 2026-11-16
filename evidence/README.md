# Validation Evidence Index

This folder supports the Weeks 12-13 validation matrix for ED Supply Decision Support version 3.1. Evidence is organized by the four required validation areas plus the two end-to-end walkthroughs.

## Primary deliverables

| Deliverable | Location | Purpose |
|---|---|---|
| Completed validation matrix | `validation/ED_Supply_Validation_Matrix_Completed.xlsx` | Connects requirements, tests, criteria, results, evidence, limitations and corrective actions |
| Validation memo | `validation/ED_Supply_Validation_Memo.docx` | Summarizes supported claims, unsupported claims, risks, priorities and suitability |
| Reproducibility record | `evidence/reproducibility_run.md` | Records a clean-environment data, model and application rerun |
| End-to-end walkthroughs | `evidence/end_to_end_walkthroughs.md` | Traces one typical scenario and one challenging scenario from decision need to action and limitations |

## Synthetic data evidence

| Evidence | Location |
|---|---|
| Data-generation and validation code | `code/generate_and_validate.py` |
| Generator settings and summary counts | `docs/pipeline_summary.json` |
| Field definitions, units and permissible values | `docs/data_dictionary.csv` |
| Validation results and violation counts | `data/final/validation_results.csv` |
| Raw generated arrivals and item assumptions | `data/raw/` |
| Final item-day and scenario tables | `data/final/` |

The checks establish internal consistency. They do not independently establish that the synthetic distributions represent a real emergency department. The matrix therefore retains a partial-pass result for independent empirical plausibility.

## Report and predictive-model evidence

| Evidence | Location |
|---|---|
| Evaluation design, leakage checks, baselines, improvements and limitations | `docs/week11_evaluation_report.md` |
| Model settings, splits, thresholds and metrics | `models/model_metadata.json` |
| Saved 2025 predictions | `data/modeling/evaluation_predictions.csv` |
| Error analysis by item and operating condition | `data/modeling/error_analysis.csv` |
| Overall predictor importance | `data/modeling/global_importance.csv` |
| Shared feature definitions | `code/model_features.py` |
| Reproducible model-training code | `code/train_predictive_models.py` |

## User-interface and user-testing evidence

| Evidence | Location |
|---|---|
| Application source | `app.py` |
| Task-based testing protocol | `docs/user_testing_protocol.md` |
| Anonymized results from two outside substitute testers | `docs/user_testing_results.md` |
| Application requirements and decision logic | `docs/app_specification.md` |

The existing user tests support basic usability and logical what-if behavior. Neither participant was an ED inventory manager, one session lacked a task-by-task record, and the tests did not fully assess explanation interpretation. These limitations remain visible in the matrix.

## Reproduction order

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/generate_and_validate.py
python code/train_predictive_models.py
streamlit run app.py
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The project uses synthetic data only. No patient-level or confidential hospital data are included.

## Revision of 15 Nov 2026: added tests and retests (matrix rows TEST-21 to TEST-33 and retests of TEST-06, 14, 15)

| Evidence | Location | Matrix rows |
|---|---|---|
| Decision-support requirements (3 decisions, do-not-use conditions) | `docs/decision_support_requirements.md` | all |
| Automated app-vs-independent-calculation results (136 checks, 0 failures) | `evidence/outputs/validation_test_results.csv`, `.json`, `validation_test_run_log.txt` | TEST-02, 06, 14, 15, 21-26 |
| KPI and filter reconciliation table | `evidence/outputs/filter_reconciliation.csv` | TEST-15 |
| Saved generated-summary and invalid-input samples | `evidence/outputs/summary_text_samples.md` | TEST-06, 14, 23, 24 |
| Baseline comparison (persistence, days of supply, risk proxy) | `evidence/outputs/baseline_comparison.csv` | TEST-27, 28 |
| Sensitivity to data-generation assumptions (5 variants) | `evidence/outputs/sensitivity_summary.csv`, `evidence/outputs/sensitivity/` | TEST-29, 30 |
| Independent benchmark comparison with sources | `docs/synthetic_data_benchmarks.md` | TEST-17 |
| Screenshots S01-S05 (Overview, Scenario Comparison, Recommended actions, invalid scenario rejected, valid scenario with summary) | `evidence/screenshots/` | TEST-06, 15, 21, 22 |
| Scripts and instructions | `validation/tests/`, `validation/README.md` | all |
| Round 2 user-test protocol (not yet run) | `docs/user_testing_round2_protocol.md` | TEST-31 |
| Original first-draft matrix | git history (commit `cd74a06`); the new workbook keeps first-draft text in each revised cell and in the "Change log" sheet | all |

Headline findings of the revision: (1) the shortage model is not robust to plausible data-generation changes (TEST-30 fails); (2) on surge days a naive persistence baseline beats the retained demand formula (TEST-27 partial); (3) the app's numbers, filters, summaries and invalid-input handling reconcile with independent calculations (0 failures in 136 checks); (4) one outside user-comprehension test is still outstanding (TEST-31).
