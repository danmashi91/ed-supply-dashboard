# Saved samples of the automatically generated scenario summaries and invalid-input messages

Captured by `validation/tests/run_validation_tests.py` from the running app.

## Demand what-if — normal-high (IVKIT, 2025-12-24, 7-day arrival forecast 1285.3 → 1485.3)

**Scenario inputs.** Seven-day arrival forecast changed from 1,285.30 to 1,485.30.

**Comparison.** Predicted seven-day demand moved from 540 to 624 units (+84 units, +15.6%).

**What drives it.** Pushing the prediction up: Seven-day arrival forecast at 1,485.30 vs. a typical 1,377.25 (+45.4 units). Pushing it down: Units used per arrival at 0.42 vs. a typical 0.81 (-579.3 units).

**Implication.** If this scenario happened, the week's stock target would need about 84 more units; the final order should still account for stock on hand, open orders, lead time and case packs.

**Trade-offs.** Ordering to the higher figure protects against running out if patient volume really rises, but ties up cash and shelf space and risks expiry if it does not. Ordering to the lower figure saves money but leaves less buffer. The estimate typically misses by about 4% of a week's demand, and by more during surges, so safety stock is still needed.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Demand what-if — normal-low (IVKIT, 2025-12-24, 7-day arrival forecast 1285.3 → 1210.3)

**Scenario inputs.** Seven-day arrival forecast changed from 1,285.30 to 1,210.30.

**Comparison.** Predicted seven-day demand moved from 540 to 508 units (-31 units, -5.8%).

**What drives it.** Pushing it down: Units used per arrival at 0.42 vs. a typical 0.81 (-472.0 units), Seven-day arrival forecast at 1,210.30 vs. a typical 1,377.25 (-70.1 units).

**Implication.** If this scenario happened, the week's stock target would need about 31 fewer units; the final order should still account for stock on hand, open orders, lead time and case packs.

**Trade-offs.** Ordering to the higher figure protects against running out if patient volume really rises, but ties up cash and shelf space and risks expiry if it does not. Ordering to the lower figure saves money but leaves less buffer. The estimate typically misses by about 4% of a week's demand, and by more during surges, so safety stock is still needed.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Demand what-if — boundary-max (IVKIT, 2025-12-24, 7-day arrival forecast 1285.3 → 1495.6)

**Scenario inputs.** Seven-day arrival forecast changed from 1,285.30 to 1,495.60.

**Comparison.** Predicted seven-day demand moved from 540 to 628 units (+88 units, +16.4%).

**What drives it.** Pushing the prediction up: Seven-day arrival forecast at 1,495.60 vs. a typical 1,377.25 (+49.7 units). Pushing it down: Units used per arrival at 0.42 vs. a typical 0.81 (-583.3 units).

**Implication.** If this scenario happened, the week's stock target would need about 88 more units; the final order should still account for stock on hand, open orders, lead time and case packs.

**Trade-offs.** Ordering to the higher figure protects against running out if patient volume really rises, but ties up cash and shelf space and risks expiry if it does not. Ordering to the lower figure saves money but leaves less buffer. The estimate typically misses by about 4% of a week's demand, and by more during surges, so safety stock is still needed.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Demand what-if — boundary-min (IVKIT, 2025-12-24, 7-day arrival forecast 1285.3 → 1078.0)

**Scenario inputs.** Seven-day arrival forecast changed from 1,285.30 to 1,078.

**Comparison.** Predicted seven-day demand moved from 540 to 453 units (-87 units, -16.1%).

**What drives it.** Pushing it down: Units used per arrival at 0.42 vs. a typical 0.81 (-420.4 units), Seven-day arrival forecast at 1,078 vs. a typical 1,377.25 (-125.7 units).

**Implication.** If this scenario happened, the week's stock target would need about 87 fewer units; the final order should still account for stock on hand, open orders, lead time and case packs.

**Trade-offs.** Ordering to the higher figure protects against running out if patient volume really rises, but ties up cash and shelf space and risks expiry if it does not. Ordering to the lower figure saves money but leaves less buffer. The estimate typically misses by about 4% of a week's demand, and by more during surges, so safety stock is still needed.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Demand what-if — invalid-high-ratio (IVKIT, 2024-01-13, 7-day arrival forecast 1495.6; one-day forecast 142.0)

**Error shown:** The seven-day arrival forecast (1,496) is 1.50× seven times tomorrow's forecast (142 × 7 = 994). That is outside the plausible band (0.75–1.40×), so the week would be inconsistent with tomorrow's forecast.

**Info shown:** No scenario prediction was produced, because a number from an inconsistent scenario would be misleading. Adjust the inputs and run the scenario again.

(No prediction table or summary was displayed.)

## Demand what-if — invalid-low-ratio (IVKIT, 2025-01-28, 7-day arrival forecast 1078.0; one-day forecast 208.8)

**Error shown:** The seven-day arrival forecast (1,078) is 0.74× seven times tomorrow's forecast (209 × 7 = 1,462). That is outside the plausible band (0.75–1.40×), so the week would be inconsistent with tomorrow's forecast.

**Info shown:** No scenario prediction was produced, because a number from an inconsistent scenario would be misleading. Adjust the inputs and run the scenario again.

(No prediction table or summary was displayed.)

## Shortage what-if — normal-lower-stock (GAUZE, 2025-12-24, closing stock 1284 → 642)

**Scenario inputs.** Days of supply changed from 6.08 to 3.14; Closing stock changed from 1,284 to 642.

**Comparison.** The estimated seven-day shortage probability moved from 1.4% to 2.7% (+1.3 percentage points).

**What drives it.** Pushing the prediction up: Seven-day arrival forecast at 1,285.30 vs. a typical 1,377.25 (+1.1 percentage points), Trailing seven-day average demand at 218.71 vs. a typical 155.57 (+1.0 percentage points). Pushing it down: Average daily arrivals, past 28 days at 186.04 vs. a typical 196.02 (-1.6 percentage points), Previous-day item demand at 256 vs. a typical 153 (-0.5 percentage points).

**Implication.** This is below the 5.6% screening threshold, so the item is not flagged. Routine review still applies.

**Trade-offs.** Acting early (an extra order, a transfer or a supplier call) lowers the risk of running out, but costs staff time, cash and shelf space, and most flags are false alarms: roughly 1 in 10 flagged items actually ran short in 2025. Not acting saves that effort but the model misses about 4 in 10 real shortages, so a low score is not a guarantee.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Shortage what-if — boundary-min-stock (GAUZE, 2025-12-24, closing stock 1284 → 0)

**Scenario inputs.** Days of supply changed from 6.08 to 0.21; Closing stock changed from 1,284 to 0.

**Comparison.** The estimated seven-day shortage probability moved from 1.4% to 7.8% (+6.4 percentage points).

**What drives it.** Pushing the prediction up: Closing stock at 0 vs. a typical 865.50 (+5.9 percentage points), Seven-day arrival forecast at 1,285.30 vs. a typical 1,377.25 (+4.2 percentage points). Pushing it down: Average daily arrivals, past 28 days at 186.04 vs. a typical 196.02 (-5.8 percentage points), Days of supply at 0.21 vs. a typical 8.45 (-3.4 percentage points).

**Implication.** This is at or above the 5.6% screening threshold, so the scenario moves the item **onto** the review list: check stock, open orders and the supplier before deciding whether to order early or escalate.

**Trade-offs.** Acting early (an extra order, a transfer or a supplier call) lowers the risk of running out, but costs staff time, cash and shelf space, and most flags are false alarms: roughly 1 in 10 flagged items actually ran short in 2025. Not acting saves that effort but the model misses about 4 in 10 real shortages, so a low score is not a guarantee.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.

## Shortage what-if — boundary-max-stock (GAUZE, 2025-12-24, closing stock 1284 → 1624)

**Scenario inputs.** Days of supply changed from 6.08 to 7.63; Closing stock changed from 1,284 to 1,624.

**Comparison.** The estimated seven-day shortage probability moved from 1.4% to 1.2% (-0.2 percentage points).

**What drives it.** Pushing the prediction up: Trailing seven-day average demand at 218.71 vs. a typical 155.57 (+0.2 percentage points), Seven-day arrival forecast at 1,285.30 vs. a typical 1,377.25 (+0.1 percentage points). Pushing it down: Closing stock at 1,624 vs. a typical 865.50 (-1.0 percentage points), Average daily arrivals, past 28 days at 186.04 vs. a typical 196.02 (-0.2 percentage points).

**Implication.** This is below the 5.6% screening threshold, so the item is not flagged. Routine review still applies.

**Trade-offs.** Acting early (an extra order, a transfer or a supplier call) lowers the risk of running out, but costs staff time, cash and shelf space, and most flags are false alarms: roughly 1 in 10 flagged items actually ran short in 2025. Not acting saves that effort but the model misses about 4 in 10 real shortages, so a low score is not a guarantee.

**Limitations.** This is a model-based scenario on synthetic data. The difference is what the model predicts under these inputs, not a proven or guaranteed effect of changing them in the real world. Inputs are kept within 2024 ranges and related fields are kept consistent, but a plausible scenario can still differ from real operations. Not an automatic purchasing or clinical decision.
