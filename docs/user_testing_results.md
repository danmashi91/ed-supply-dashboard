# User Acceptance Testing Results

## Test context

- **Test date:** November 2, 2026
- **Participant:** Adult outside the project group
- **Decision-maker status:** Substitute participant, not an emergency-department inventory manager
- **Data shown:** Synthetic data only
- **Objective:** Evaluate the responsiveness, usability and practical usefulness of the model-based what-if analysis

The participant tested two demand scenarios by changing the seven-day patient-arrival forecast while the remaining inputs stayed fixed.

## Task-by-task record

| Rubric task | Participant action or response | Outcome | Error, confusion or missing information |
|---|---|---|---|
| Interpret the baseline prediction | Used the displayed baseline of 1,285.30 expected arrivals and 540 predicted units of seven-day demand as the reference for both comparisons. | The participant treated 540 units as the baseline demand estimate for inventory planning. | No misunderstanding reported. |
| Identify the main factor contributing to the prediction | Identified the seven-day patient-arrival forecast as the factor changed in the test while the one-day forecast, current arrivals and units used per arrival remained fixed. | The participant understood that expected patient volume affected predicted inventory demand. | The test did not separately assess whether the participant could distinguish overall predictor importance from contributions to one prediction. |
| Create and compare Scenario 1 | Increased the seven-day arrival forecast from 1,285.30 to 1,485.30. Predicted demand increased from 540 to 624 units, a difference of 84 units. | Completed. The prediction moved in the expected direction. | No error reported. |
| Create and compare Scenario 2 | Decreased the seven-day arrival forecast from 1,285.30 to 1,210.30. Predicted demand decreased from 540 to 508 units, a difference of 31 units. | Completed. The prediction moved in the expected direction. | No error reported. |
| Explain decision usefulness | Reported that the demand estimates could support hospital inventory planning by showing how expected patient volume changes inventory requirements. | The participant found the side-by-side comparison useful and understandable. | Requested more explicit inventory recommendations. |
| Identify unclear or missing information | Reported that only one input was editable, which limited the range of operational assumptions that could be tested. | The participant found navigation easy but wanted additional adjustable inputs. | The application needs clearer wording about why some predictors are fixed, derived or external and which operational inputs can reasonably change. |

## Participant feedback

### Positive findings

1. The application responded consistently when the seven-day arrival forecast increased and decreased.
2. Predicted demand moved in the expected direction in both scenarios.
3. The side-by-side comparison clearly showed the baseline, modified scenario and difference.
4. The participant found the application easy to navigate.
5. The results provided useful demand estimates for inventory planning.

### Improvement requests

1. Allow additional reasonable inputs to be adjusted so users can test more operational assumptions.
2. Provide more explicit recommendations for inventory managers.

## Changes linked to the findings

- The Predictive Models page includes a **Recommended actions** table that identifies items requiring immediate review, items that may need an order during the week and items requiring no action. Suggested quantities use whole case packs. These remain model-based recommendations that require managerial review.
- Scenario inputs are labelled as external conditions, fixed observations, item characteristics or derived values. The application locks values that the inventory manager cannot reasonably change or that must remain internally consistent.
- The shortage what-if model provides an operational planning input through editable closing stock. The demand model keeps units used per arrival and current arrivals fixed because they are item characteristics or observed values, while the external arrival forecast remains editable for planning scenarios.

The request for more editable fields was therefore addressed through clearer control labels and the separate shortage scenario rather than by unlocking fixed historical values or item characteristics.

## Overall assessment

Both scenarios passed the functional responsiveness check. The participant concluded that the what-if functionality responded as expected and could support inventory planning. The test supports usability and logical scenario behavior, but it does not establish real-world predictive accuracy.

## Limitations

- The participant was a substitute rather than an ED inventory manager.
- The test used one participant.
- Both scenarios evaluated the demand model and changed the same external predictor.
- The test did not separately assess interpretation of overall predictor importance versus individual prediction contributions.
- Synthetic data limit conclusions about real hospital performance.
- Held-out evaluation results provide the formal predictive-performance evidence; the user test evaluates usefulness and comprehension.
