# End-to-End Decision Walkthroughs

These walkthroughs connect the decision need, input data, report or model output, interface presentation, user interpretation, proposed action and limitations. They support matrix tests TEST-19 and TEST-20.

## Walkthrough 1 Typical replenishment review

**Decision need.** The ED inventory manager reviews expected supply use and decides whether an item needs replenishment attention for the next seven days.

**Input data.** The tested baseline used a seven-day arrival forecast of 1,285.30 patients and an item use rate of 0.42 units per arrival. The participant then entered a plausible higher-arrival scenario of 1,485.30 while the fixed fields remained unchanged.

**Report and model output.** The retained calculation is the seven-day arrival forecast multiplied by units per arrival. The baseline calculation is 1,285.30 × 0.42 = 539.826, displayed as 540 units. The higher-arrival scenario is 1,485.30 × 0.42 = 623.826, displayed as 624 units, a difference of 84 units.

**UI presentation.** The Predictive Models page shows the baseline, modified scenario and difference side by side. Inputs are labelled as external, fixed, derived or item characteristics. The Recommended actions table places each selected item into Review now, Order this week or No action needed and rounds any suggested quantity to whole case packs.

**User interpretation.** Tester B correctly treated 540 units as the baseline and understood that the higher patient forecast increased estimated item demand to 624 units.

**Proposed action.** The manager can use the increase as a reason to review current stock and the suggested case-pack quantity before placing or adjusting an order.

**Limitations.** The participant was a substitute rather than an ED inventory manager. The suggested quantity does not include safety stock, open orders, full lead-time variation, storage capacity, cost or expiration risk. The result is a model-based planning scenario, not a guaranteed effect or automatic purchase instruction.

**Conclusion.** The application adequately supports a classroom demonstration of a typical replenishment review. For operational use, it would support the decision only after real-data validation and the addition of hospital-approved inventory rules.

## Walkthrough 2 Surge and shortage risk

**Decision need.** The ED inventory manager must identify items that may run short during a surge while avoiding unnecessary emergency orders and excess stock.

**Input data.** Relevant inputs include the seven-day arrival forecast, recent demand, closing stock, days of supply, supplier status and surge condition. The shortage model uses values available at the end of the review day and predicts whether any backorder will occur during the next seven days.

**Report and model output.** The retained shortage model achieved ROC-AUC 0.9193 and average precision 0.2518 on the synthetic 2025 holdout. At the selected 5.61% threshold, it flags 5.06% of item-days, captures 58.33% of positive cases and has 9.66% precision. Demand error is materially worse during surges: 11.1% WAPE and average underprediction of about 167 units per item-week. Shortage recall is 84.6% during surges but 27.3% outside surges.

**UI presentation.** The application presents shortage probability, the screening threshold, overall importance, individual contributions, a stock what-if control, a side-by-side comparison and a recommended review category.

**User interpretation.** A high-risk flag means the item should be reviewed first. It does not mean a shortage is certain. Because precision is low, most alerts are false positives; because recall is incomplete, the absence of an alert does not guarantee adequate stock.

**Proposed action.** The manager should verify physical stock, open orders, supplier timing and local safety-stock rules, then decide whether to expedite, order routinely or continue monitoring. During a surge, the manager should test a higher patient forecast and apply additional managerial caution because the demand estimate tends to underpredict.

**Limitations and conflict.** The tool does not model safety stock, open-order timing, expiration, storage capacity, supplier cost or the cost difference between a false alert and a missed shortage. Only 24 shortage-positive test rows exist, concentrated in three of eight items. No ED inventory manager completed this full challenging walkthrough.

**Conclusion.** The application supports the challenging decision only under stated classroom conditions. It is not adequate for autonomous ordering, a controlled pilot or operational use until the missing operational controls, target-user testing and authorized real-data validation are completed.
