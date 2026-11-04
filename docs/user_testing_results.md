# User Acceptance Testing Results

Two people outside the project group tested the deployed application independently. Both are **substitutes** for the intended decision-maker: neither is an emergency-department inventory manager. Only synthetic data was shown; no confidential or patient information was involved. Participants are described by general role only.

| | Tester A | Tester B |
|---|---|---|
| Background | Saint Louis University peer outside the project group | Adult outside the project group |
| Date | November 2, 2026 | November 2, 2026 |
| Arranged by | Abubakar Sani | Monica Hall |
| Format | Explored all pages, then gave written feedback | Completed demand what-if tasks and reported results and feedback |

## Tester A

### What the participant did

Tester A explored the deployed application, including the descriptive pages, the predictive models and the what-if analysis, and then gave written feedback. A task-by-task observation record was not captured for this session.

### Positive findings

1. Focusing on two decisions, predicting demand and identifying possible shortages, was clear and relevant.
2. The dashboard was well organized; the trend and spending charts helped explain inventory patterns.
3. Scenario comparison and what-if analysis made the application more useful for decision-making.

### Problems and requests

1. **No direct recommendations.** The participant asked for simple recommendations based on the predictions, such as which items need immediate attention and when additional inventory should be ordered.
2. **Crowded layout.** Some charts and sections looked crowded; better spacing and readability would make the dashboard easier to navigate.

## Tester B

### Task-by-task record

| Rubric task | Participant action or response | Outcome | Error, confusion or missing information |
|---|---|---|---|
| Interpret the baseline prediction | Used the displayed baseline of 1,285.30 expected arrivals and 540 predicted units of seven-day demand as the reference for both comparisons. | Treated 540 units as the baseline demand estimate for inventory planning. | No misunderstanding reported. |
| Identify the main factor contributing to the prediction | Identified the seven-day patient-arrival forecast as the factor changed, while the one-day forecast, current arrivals and units used per arrival stayed fixed. | Understood that expected patient volume affected predicted demand. | The test did not separately assess whether the participant could distinguish overall predictor importance from contributions to one prediction. |
| Create and compare Scenario 1 | Increased the seven-day arrival forecast from 1,285.30 to 1,485.30. Predicted demand rose from 540 to 624 units (+84). | Completed; the prediction moved in the expected direction. | No error reported. |
| Create and compare Scenario 2 | Decreased the seven-day arrival forecast from 1,285.30 to 1,210.30. Predicted demand fell from 540 to 508 units (−31). | Completed; the prediction moved in the expected direction. | No error reported. |
| Explain decision usefulness | Reported that the estimates could support inventory planning by showing how expected patient volume changes requirements. | Found the side-by-side comparison useful and understandable. | Requested more explicit inventory recommendations. |
| Identify unclear or missing information | Reported that only one input was editable in the demand scenario, limiting the operational assumptions that could be tested. | Found navigation easy but wanted more adjustable inputs. | Needed clearer wording on why some predictors are fixed, derived or external. |

### Positive findings

1. The application responded consistently when the seven-day arrival forecast increased and decreased, in the expected direction both times.
2. The side-by-side comparison clearly showed the baseline, the modified scenario and the difference.
3. The application was easy to navigate, and the estimates were useful for inventory planning.

### Problems and requests

1. **Limited editable inputs** in the demand scenario.
2. **More explicit recommendations** for inventory managers.

## Shared finding

Both testers, independently, asked for **clearer inventory recommendations** rather than predicted values alone. This was the highest-priority change.

## Changes linked to the findings

| Finding | Tester | Change made |
|---|---|---|
| Wanted recommendations on which items need attention and when to order | A and B | The Predictive Models page now opens with a **Recommended actions** table classifying each selected item as *Review now*, *Order this week* or *No action needed*, with a suggested order in whole case packs. It is labelled as a model-based suggestion for managerial review, not an automatic order. |
| Crowded charts and sections | A | Major sections on the Predictive Models page are separated by dividers, and long evaluation tables are folded into expandable sections while key conclusions stay visible. |
| Only one editable input; unclear why others are fixed | B | Scenario inputs are labelled as external conditions, fixed observations, item characteristics or derived values, with an explanation of how input checks work. The shortage what-if offers an operational planning input through editable closing stock. Fixed historical values and item characteristics stay locked because the manager cannot change them at prediction time. |

**Not changed.** A full visual redesign of the other pages, and additional operational controls (open-order timing, lead-time assumptions, safety-stock targets), are future work that should follow requirements from a real ED inventory manager.

## Overall assessment

Both testers found the what-if analysis useful for decision-making, and Tester B confirmed that demand predictions respond in the expected direction to patient volume. The tests support usability and logical scenario behaviour; they do not establish real-world predictive accuracy, which is evidenced by the held-out 2025 evaluation.

## Limitations

- Both participants were substitutes rather than ED inventory managers.
- Tester A's session has written feedback but no task-by-task observation record.
- Tester B's two scenarios both evaluated the demand model and changed the same external predictor.
- Neither test separately assessed interpretation of overall predictor importance versus individual prediction contributions.
- Synthetic data limit conclusions about real hospital performance.
