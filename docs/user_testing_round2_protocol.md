# User Testing Round 2: Protocol and Record Form (TEST-31)

**Status: NOT YET RUN.** This needs a person outside the group, preferably an ED inventory or hospital supply-chain professional (otherwise a knowledgeable substitute). It covers what the Week 11 tests did not: explanation interpretation, the active-filter caption, recognizing a rejected scenario, the recommended-actions table and the warnings. Show synthetic data only; record only a general role description.

## Setup
Deployed app: https://ed-supply-dashboard.streamlit.app/ (version 3.1). Do not coach. Note any help requested. Pass = completes the task without help and states the expected conclusion.

## Participant
Relationship to intended users (circle): ED inventory manager / hospital supply-chain staff / nurse or ED staff / other knowledgeable substitute / general adult. Date: ____ Facilitator: ____ Observer: ____

## Tasks

| # | Task (read aloud) | Expected outcome | Completed? | Help needed | Interpretation error or confusion |
|---|---|---|---|---|---|
| 1 | On the Overview page set the date range to March 2025 and choose only Sterile gauze pack. Say how many days and item rows are included and what the arrival cards cover. | Reads "31 days, 1 item, 31 rows" from the sidebar caption; notes arrival cards ignore the item filter | | | |
| 2 | On Predictive Models, read the Recommended actions table. Which item should be reviewed first and what does the suggested order mean? | Names the top category; states that it is a suggestion for manager review, in whole cases, not an order | | | |
| 3 | In the demand what-if, raise the seven-day arrival forecast by about 200 and run it. What happened to demand and why? | Reports about +84 units (IV start kit); says the model predicts this under the scenario, not a guaranteed effect | | | |
| 4 | Look at the contribution chart. Which factor caused the increase? Can you change that factor in real life? | Distinguishes the changed forecast from fixed item scale; does not claim the bars are causes | | | |
| 5 | Explain the difference between "overall importance" and "individual contributions". | States overall = across the whole test year; individual = one prediction for one item and date | | | |
| 6 | Choose date 2024-01-13 (IV start kit) and set the seven-day forecast to the maximum. Run it. What does the app tell you and what would you do? | Recognizes that the scenario was rejected and no prediction was produced; adjusts the input | | | |
| 7 | Switch to the shortage model. What does a flagged item mean, how often is a flag right, and how many shortages are caught? | "Check this item"; about 1 in 10; about 6 in 10 | | | |
| 8 | When would you NOT use these predictions? | Mentions synthetic data, surges, rare shortages, not a purchase order | | | |

## Debrief questions
1. Which part was hardest to understand? 2. What would you want that is missing? 3. Would you trust the numbers for real purchasing? Why or why not?

## After the session
Save the completed form as `docs/user_testing_round2_results.md`. Update matrix rows TEST-04, 11, 13, 20 and 31 and the Change log. Record any resulting application change.
