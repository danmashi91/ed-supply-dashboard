# Seven-Minute Week 10 Demonstration Script

Target length: 6:40–7:00. Both members should say their names and describe their work. Replace the bracketed deployment link after publishing.

| Time | Speaker | Screen/action | Suggested narration |
|---|---|---|---|
| 0:00–0:35 | Monica | Title/Overview | “We are Monica Hall and Abubakar Sani. Our ED Supply Decision Support tool is for an emergency-department inventory manager deciding which routine supplies need attention, how much demand to expect, and where shortage risk may require review.” |
| 0:35–1:15 | Abubakar | Overview filters and KPI cards | Explain the two-year synthetic dataset, eight consumables, date/item filters, descriptive KPIs, and risk queue. State that these establish operational context but Week 10 adds forward-looking KPIs. |
| 1:15–1:50 | Abubakar | Item Explorer/Trends | Briefly show one item and how inventory, demand, receipts, supplier events, and order spend support investigation. Do not spend longer than 35 seconds here. |
| 1:50–2:35 | Monica | Predictive Models → Seven-day demand | Define KPI 1 as predicted units requested during the next seven days. Explain its replenishment and financial relevance. Open "Predictors and why they are relevant" and name units per arrival, recent demand, arrivals, calendar and surge. State why gradient boosting captures nonlinear patterns, and that candidates were compared on 2024 data only. |
| 2:35–3:10 | Monica | Demand validation and overall importance | Point to WAPE 3.84% and say plainly that the simple benchmark (arrival forecast × units per arrival) scores 3.79%: expected, because the synthetic data was built from that formula. In the importance chart, explain that arrival forecasts rank low because recent demand already reflects arrival levels, not because patients don't matter. Overall importance = model reliance, not causation. |
| 3:10–4:05 | Monica | Demand what-if | Select an item/date and show the baseline prediction, its contribution chart, and the actual outcome. Raise the seven-day arrival forecast and trailing demand, run the scenario, and point to baseline, updated prediction, change, scenario contributions, and automatic summary. State that local contributions explain one prediction and may not sum because of interactions. |
| 4:05–4:50 | Abubakar | Switch to shortage model | Define KPI 2 as the probability of any backorder in the next seven days. Explain relevance to review prioritization and supplier escalation. Name the predictors: coverage (days of supply vs. lead time), closing stock vs. recent demand, demand trend, arrival pressure, surge, supplier events. Explain logistic regression with class weighting plus calibration, so a stated 3% means roughly 3%. |
| 4:50–5:30 | Abubakar | Shortage validation and importance | Show ROC-AUC 0.915 and that, at the 7.3% screening threshold, it flags 8.8% of item-days and catches 75% of shortages. Disclose precision 7.1%, only 24 positive test cases, and likely false alerts. Point out closing stock as the top predictor. Say it is a screening tool, not an automatic ordering rule. |
| 5:30–6:15 | Abubakar | Shortage what-if | Cut closing stock (e.g. to about a third) and set supplier event to "delayed". Run the scenario: probability rises above the screening threshold. Compare baseline with updated probability, read the summary, and explain direction, magnitude, practical meaning, and the non-causal warning. |
| 6:15–6:40 | Monica | Data & Assumptions | Show 13/13 validation checks, synthetic-data disclosure, time coverage, and downloadable filtered table. Mention chronological training/testing and reproducible scripts. |
| 6:40–7:00 | Both | README/repository | Summarize contributions. Confirm that app, data, models, code, setup instructions, progress report, contribution summary, and video link are accessible at `[REPOSITORY LINK]`. |

## Recording checklist

- Use browser zoom that keeps KPI metrics and chart labels readable.
- Open the deployed app before recording and preselect one stable example for each model.
- Rehearse both what-if inputs so the automatic summary appears without delay.
- Say “predictive association, not causation” at least once.
- State the shortage model's low precision, the demand benchmark result, and the synthetic-data limitation aloud.
- Show both baseline and updated prediction for both models.
- Confirm the finished video is no longer than seven minutes and that the instructor can open it.

