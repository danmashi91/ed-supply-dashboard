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
