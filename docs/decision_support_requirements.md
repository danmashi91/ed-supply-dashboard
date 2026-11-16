# Decision-Support Requirements (Weeks 12-13)

Application: ED Supply Decision Support, version 3.1 (synthetic data v1.1-2026-10-19).
Retained KPIs (unchanged from Week 10/11): **KPI 1** predicted seven-day item demand; **KPI 2** probability of any backorder in the next seven days. The eight descriptive Overview KPIs (arrivals, arrival-forecast WAPE, surge days, fill rate, backorder days, high-risk item-days, order spend) are retained as supporting monitoring KPIs.

**Intended decision-maker:** an emergency-department (ED) inventory manager who orders eight routine consumables for one ED.

**Requirement sources.** Source labels follow the assignment: *DM* = intended decision-maker, *KS* = knowledgeable substitute (the two outside testers of 2 Nov 2026, neither an ED inventory manager), *PS* = published source, *PTA* = project-team assumption. No requirement comes directly from an ED inventory manager; the project-team assumptions are listed as such and are the main unvalidated element.

Acceptance criteria below were written on 15 Nov 2026. Where the evidence was gathered earlier (Weeks 10-11), the criterion was written afterwards and has not been altered to suit the results (see the matrix "Change log" sheet).

---

## D1. Weekly replenishment review (REQ-01)

| Field | Requirement |
|---|---|
| Decision | For each item, decide whether the stock position (on hand plus open orders) covers next week's expected use, and if not, how many whole cases to order. |
| Frequency and constraints | Twice weekly at the Monday and Thursday review days (data generator rule); orders in whole case packs; supplier lead times 2-5 days; the manager has the unit cost but no safety-stock policy is encoded in the tool. |
| Information required | KPI 1 predicted seven-day demand; current inventory position; case pack; scenario low/base/high demand; 7-day arrival forecast. |
| How the app should help | Show baseline demand next to a user-changed arrival scenario and the difference (units); list each item as *Review now*, *Order this week* or *No action needed* with a suggested whole-case quantity; display the typical error (about 4% of weekly demand) and that it is larger in surges. |
| Consequence of wrong information | Under-estimate: shortage and costly emergency purchase. Over-estimate: cash tied up, shelf space, expiry. Misleading direction (e.g. higher arrivals lowering demand) destroys trust, so this was explicitly tested. |
| Acceptance criteria | (a) displayed demand equals 7-day arrival forecast x use rate after rounding (TEST-02, 05, 23); (b) a higher/lower scenario moves demand in the right direction and shows baseline, scenario and signed difference (TEST-04); (c) suggested orders are whole cases and equal an independent calculation (TEST-22); (d) retained method beats a persistence baseline overall and on surge days (TEST-27); (e) result survives data-generation changes (TEST-29). |
| Source | PTA (documented in `docs/app_specification.md`); output format requested by KS testers (recommended actions). |

## D2. Shortage screening and escalation (REQ-02)

| Field | Requirement |
|---|---|
| Decision | Decide which items to check first for a possible stock-out in the next seven days, and whether to expedite or escalate to the supplier. |
| Frequency and constraints | Daily, limited manager time; cannot check every item every day; a false alarm costs staff time, a miss risks patient-care disruption. |
| Information required | KPI 2 shortage probability; the screening threshold; days of supply; supplier events; surge flag; performance numbers (precision and recall). |
| How the app should help | Rank items by shortage probability; show the threshold and what a flag means (check this item, not order now); show the error rates (about 1 in 10 flags real; about 6 in 10 shortages caught) and which items have no shortage evidence. |
| Consequence of wrong information | False alerts cause alert fatigue; missed shortages (about 42% at the threshold) cause service disruption; presenting a probability as a guarantee would mislead. |
| Acceptance criteria | (a) model beats a constant-rate baseline and simple stock rules at equal alert volume (TEST-08, 28); (b) no use of information unavailable at prediction time (TEST-08); (c) alert share, precision, recall, surge and item breakdown reported (TEST-09, 10); (d) skill is robust to plausible changes in generator assumptions (TEST-30); (e) warnings about rarity, false alarms and unsupported items displayed (TEST-26). |
| Source | PTA (`docs/week10_predictive_specification.md`). |

## D3. Monitoring and interpreting current status (REQ-03)

| Field | Requirement |
|---|---|
| Decision | Decide where to look today: which category or item is under pressure, whether arrivals are in a surge, and whether the explanation of a prediction can be trusted. |
| Frequency and constraints | Daily; filters restrict date range, category, item and risk threshold; arrival KPIs do not depend on item filters. |
| Information required | Eight Overview KPI cards; filtered tables; the active population (days, items, rows); explanations (overall importance vs one-prediction contributions). |
| How the app should help | Cards show value, unit and period; sidebar caption states the active filter population on every page; explanations are labelled as approximate and non-causal; empty selections give a message, not zeros. |
| Consequence of wrong information | A hidden filter mismatch shows inconsistent numbers across pages; a causal reading of contribution bars could prompt changing a fixed item characteristic. |
| Acceptance criteria | (a) every Overview and Item Explorer KPI equals an independent calculation under normal, boundary and empty filters (TEST-15, 25); (b) the active-population caption matches on all pages (TEST-15); (c) generated summary contains figures equal to displayed values, includes limitations and avoids causal wording (TEST-14, 24); (d) an outside user interprets explanations and filters without error (TEST-13, 31, still untested). |
| Source | PS/PTA (course requirement for non-causal explanations; team design); KS feedback on layout. |

## Cross-cutting requirements

* **REQ-04 Data fitness.** Synthetic data must be structurally valid, reproducible, documented and free of confidential records (TEST-01, 07, 16, 17, 18). Empirical realism is only partly supported; see `docs/synthetic_data_benchmarks.md`.
* **REQ-05 End-to-end decisions.** A typical and a challenging scenario must be traceable from need to action with limitations stated (TEST-19, 20).

## Do-not-use conditions (documented for TEST-32)

1. Do not use either model for real purchasing, clinical or patient-safety decisions; data are synthetic.
2. Do not read the shortage probability as a guarantee; at the chosen threshold about 9 of 10 flags are false and about 4 of 10 shortages are missed.
3. Do not rely on the shortage score for items that had no shortage in the test year (five of eight items); subgroup estimates are unstable.
4. Do not rely on the baseline demand estimate during a surge; it under-predicted by about 167 units on surge rows and a persistence baseline did better (TEST-27). Test a higher arrival forecast and add safety stock.
5. Do not treat the shortage model's skill (ROC-AUC 0.919) as stable; it fell to 0.66-0.80 under four plausible alternative generator settings (TEST-30). It must be retrained and re-evaluated on any new data.
6. Do not use the screening threshold (5.61%) with new data without re-selecting it; in one variant it flagged 99.3% of item-days.
7. Do not use scenarios outside the 2024 development ranges or with a seven-day forecast outside 0.75-1.40 times seven times the one-day forecast (the app blocks these).
8. Do not use suggested order quantities as purchase orders: safety stock, open-order timing, lead-time variability, expiry, storage and cost are not modelled.
