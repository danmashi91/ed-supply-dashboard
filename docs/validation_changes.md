# Changes Made in Response to Validation Findings

## Application code
`app.py`, the models and the data are **unchanged** in this revision (version 3.1). The new tests found no defect in the displayed numbers, filters, summaries or invalid-input handling, so no fix was required for those. Two findings call for changes that the group has not yet made:

| Finding | Matrix row | Proposed change (not yet implemented) |
|---|---|---|
| Shortage-model skill is unstable across plausible data-generation changes (ROC-AUC 0.919 -> 0.67-0.80; threshold alert share 0.24-99.3%) | TEST-30, 32 | Add a caution to the shortage section of the Predictive Models page; report results over several seeds; re-choose the threshold per data set |
| On surge days a persistence baseline has lower demand error and bias than the arrival formula | TEST-27 | Show a persistence cross-check or a surge override next to the baseline demand estimate |
| "What drives it" sentence highlights a fixed item characteristic (-579 units) | TEST-13 | Limit the sentence to predictors the user changed, or label fixed item scale as not a driver of the change |

## Validation materials
New scripts, outputs, screenshots, requirements, benchmark comparison and a Round 2 protocol were added (see `evidence/README.md`). The first-draft matrix is preserved in git history (commit `cd74a06`) and in each revised cell and the matrix "Change log" sheet.
