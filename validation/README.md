# Validation materials: how to repeat the tests

All commands are run from the repository root. Python 3.10 or later (tested on 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r validation/requirements-validation.txt     # adds openpyxl, python-docx, playwright (screenshots only)
python code/generate_and_validate.py                      # regenerate data (seed 20261009); git should show no changes
python code/train_predictive_models.py                    # retrain; metrics must match models/model_metadata.json

python validation/tests/run_validation_tests.py           # about 3-5 min; 136 automated checks against the running app logic
python validation/tests/baseline_comparison.py            # naive and stock-rule baselines (TEST-27, TEST-28)
python validation/tests/run_sensitivity.py "$(which python)"   # 5 generator variants, about 7 min in parallel batches (TEST-29, TEST-30)
# optional screenshots: start the app (streamlit run app.py --server.port 8599) then
python validation/tests/capture_screenshots.py            # needs Google Chrome; writes evidence/screenshots/
```

| Script | Matrix rows | Output |
|---|---|---|
| `validation/tests/run_validation_tests.py` | TEST-02, 06, 14, 15, 21-26 | `evidence/outputs/validation_test_results.csv/.json`, `filter_reconciliation.csv`, `summary_text_samples.md` |
| `validation/tests/baseline_comparison.py` | TEST-27, 28 | `evidence/outputs/baseline_comparison.csv` |
| `validation/tests/run_sensitivity.py` | TEST-29, 30 | `evidence/outputs/sensitivity_summary.csv`, `evidence/outputs/sensitivity/*.json` |
| `validation/tests/capture_screenshots.py` | TEST-06, 15, 21, 22 | `evidence/screenshots/S01-S05` |
| `validation/tests/apptest_helpers.py` | (helper) | Opens one page of the app headlessly with Streamlit AppTest |

How the app tests work: `streamlit.testing.v1.AppTest` runs the real `app.py` in memory. Because the app uses `st.navigation`, the helper writes a temporary copy of `app.py` with the requested page marked as default, runs it, then deletes the copy. Displayed values are compared with pandas calculations made directly from the CSV files, not with the app's own functions.

Notes
* The sensitivity script patches generator constants in a temporary copy of the repository; the committed data and models are never modified.
* Results can differ slightly across scikit-learn versions; the pinned version is 1.7.2.
* `docs/user_testing_round2_protocol.md` is the human test that these scripts cannot replace (TEST-31).
