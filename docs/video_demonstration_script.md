# Seven-Minute Week 10 Demonstration Script

Target length: 6:40–7:00. Both members should say their names and describe their work. Replace the bracketed deployment link after publishing.

| Time | Speaker | Screen/action | Suggested narration |
|---|---|---|---|
| 0:00–0:35 | Monica | Title/Overview | “We are Monica Hall and Abubakar Sani. Our ED Supply Decision Support tool is for an emergency-department inventory manager deciding which routine supplies need attention, how much demand to expect, and where shortage risk may require review.” |
| 0:35–1:15 | Abubakar | Overview filters and KPI cards | Explain the two-year synthetic dataset, eight consumables, date/item filters, descriptive KPIs, and risk queue. State that these establish operational context but Week 10 adds forward-looking KPIs. |
| 1:15–1:50 | Abubakar | Item Explorer/Trends | Briefly show one item and how inventory, demand, receipts, supplier events, and order spend support investigation. Do not spend longer than 35 seconds here. |
| 1:50–2:35 | Monica | Predictive Models → Seven-day demand | Define KPI 1 as predicted units requested during the next seven days. Explain its replenishment and financial relevance. Identify recent demand, arrivals, item, calendar, and surge predictors. State why gradient boosting captures nonlinear patterns. |
| 2:35–3:10 | Monica | Demand validation and overall importance | Point to WAPE 3.85%, MAE 58.7 units, RMSE 95.4, and R² 0.995 on the 2025 holdout. Explain that permutation importance is overall model reliance, not causation, and that synthetic structure may make performance optimistic. |
| 3:10–4:05 | Monica | Demand what-if | Select an item/date, identify the baseline, change arrival forecast or recent demand, and run the scenario. Point to baseline, updated prediction, change, local contribution chart, and automatic summary. State that local contributions explain one prediction and may not sum because of interactions. |
| 4:05–4:50 | Abubakar | Switch to shortage model | Define KPI 2 as the probability of any backorder in the next seven days. Explain relevance to review prioritization and supplier escalation. Identify inventory, days of supply, open orders, demand, lead time, surge, and supplier-event predictors. Explain the class-weighted logistic regression choice. |
| 4:50–5:30 | Abubakar | Shortage validation and importance | Show ROC-AUC 0.871 and recall 70.8%. Explicitly disclose precision 4.25%, 24 positive test cases, and likely false alerts. Say it is a screening tool rather than an automatic ordering rule. |
| 5:30–6:15 | Abubakar | Shortage what-if | Modify days of supply, inventory, lead time, or supplier event. Run the scenario and compare baseline with updated probability. Explain direction, magnitude, practical meaning, and the non-causal warning. |
| 6:15–6:40 | Monica | Data & Assumptions | Show 13/13 validation checks, synthetic-data disclosure, time coverage, and downloadable filtered table. Mention chronological training/testing and reproducible scripts. |
| 6:40–7:00 | Both | README/repository | Summarize contributions. Confirm that app, data, models, code, setup instructions, progress report, contribution summary, and video link are accessible at `[REPOSITORY LINK]`. |

## Recording checklist

- Use browser zoom that keeps KPI metrics and chart labels readable.
- Open the deployed app before recording and preselect one stable example for each model.
- Rehearse both what-if inputs so the automatic summary appears without delay.
- Say “predictive association, not causation” at least once.
- State the shortage model's low precision and synthetic-data limitation aloud.
- Show both baseline and updated prediction for both models.
- Confirm the finished video is no longer than seven minutes and that the instructor can open it.

