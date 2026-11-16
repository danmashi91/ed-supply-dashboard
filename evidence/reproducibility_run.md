# Clean Environment Reproducibility Record

## Test information

- Test ID: TEST-16
- Test date: November 15, 2026
- Tester: Monica Hall
- Repository commit tested: `34d6c54df45055a30e8ceed5622bbadba2538fc4`
- Environment: newly created Python 3.12 virtual environment
- Dependency source: `requirements.txt`

## Procedure

The following sequence was run from the repository root:

```bash
python3 -m venv .validation_venv
.validation_venv/bin/pip install -r requirements.txt
.validation_venv/bin/python code/generate_and_validate.py
.validation_venv/bin/python code/train_predictive_models.py
```

The application was then loaded with Streamlit's automated `AppTest` runner. The test completed with zero application exceptions.

## Data-pipeline result

| Check | Reproduced result |
|---|---:|
| Random seed | 20261009 |
| Date coverage | 2024-01-01 to 2025-12-31 |
| Arrival rows | 731 |
| Item master rows | 8 |
| Final item-day rows | 5,848 |
| Final item-day columns | 28 |
| Scenario rows | 17,544 |
| Scenario columns | 10 |
| Validation checks passed | 13 of 13 |
| Arrival forecast WAPE | 9.35% |
| Fill-rate proxy | 99.88% |
| High-risk item-days | 223 |

After generation, Git reported no changes to the committed generated data or summary files. This confirms that the fixed seed and documented pipeline reproduced the repository outputs.

## Model-training result

| KPI | Reproduced final result |
|---|---:|
| Seven-day demand test WAPE | 3.79% |
| Seven-day demand MAE | 57.769 units |
| Shortage test ROC-AUC | 0.9193 |
| Shortage test average precision | 0.2518 |
| Shortage Brier score | 0.0075 |
| Shortage screening threshold | 5.61% |
| Shortage precision at threshold | 9.66% |
| Shortage recall at threshold | 58.33% |

The retained models, candidate comparisons and predictor-importance table matched the committed metadata and evaluation report. A joblib warning reported that physical CPU cores could not be detected in the container and logical cores were used instead. The run completed successfully and results were unchanged.

## Application-load result

- Streamlit application file: `app.py`
- Automated result: zero exceptions on initial application run
- Manual deployment: `https://ed-supply-dashboard.streamlit.app/`

## Conclusion and limitation

Status: **Pass for technical reproducibility.** The documented code, seed, dependencies and run order reproduce the stored data and model results in a fresh environment.

This test does not validate real-world accuracy. It also does not prove identical behavior on every operating system or future dependency release. The test should be repeated after any change to data generation, dependencies, features or model logic.
