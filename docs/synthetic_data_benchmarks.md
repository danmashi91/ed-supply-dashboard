# Synthetic Data vs Independent Sources (supports TEST-17)

Purpose: compare the synthetic ED data (v1.1-2026-10-19) with published information. Agreement with the generator's own rules is internal consistency only. Only the comparisons below are independent, and each has limits. Searches were run on 15 Nov 2026; sources should be re-checked before citation in a paper.

## Calculations from the synthetic data

| Quantity | Synthetic value | How computed |
|---|---|---|
| Mean arrivals per day | 193.8 (about 70,700 per year) | `generated_ed_arrivals.csv`, 731 days |
| Share of weekly arrivals by day | Mon 14.96%, Tue 14.89%, Wed 14.63%, Thu 14.53%, Fri 14.63%, Sat 13.30%, Sun 13.07% | day totals / total |
| Mean arrivals on flagged holidays vs other days | 181.8 vs 193.9 (about -6%; generator uses 0.88 on three holidays) | holiday_flag |
| Backorder rate | 0.33% of item-days (19 of 5,848); fill-rate proxy 99.88% | `inventory_decision_daily.csv` |

## Comparison with published sources

| Variable | Published reference (paraphrased) | Source | Agreement | Limits |
|---|---|---|---|---|
| Annual ED volume | A 389-bed regional medical centre reports about 80,000 adult ED visits a year; the US had about 155 million ED visits in 2022. | Ebrahimihoor et al., J Community Hosp Intern Med Perspect 2023 (https://pmc.ncbi.nlm.nih.gov/articles/PMC10589040); NCHS Data Brief 503, 2024 (https://stacks.cdc.gov/view/cdc/159284/cdc_159284_DS1.pdf) | Plausible: 70,700 per year is a mid-size ED | Single-site; one example is not a distribution |
| Day-of-week pattern | Monday is the busiest day (15.7% of weekly visits at the regional centre, 16.2% in NHAMCS, 15.5% in TriNetX) and weekends have the least volume. A Lubbock hospital found Monday busiest (235.6 per day) and Saturday lowest (201.9). | Same PMC article; Jaroudi et al., SW Respir Crit Care Chron 2019 (https://pulmonarychronicles.com/index.php/pulmonarychronicles/article/download/513/1134/2950) | Direction agrees (Monday highest, weekend lowest); synthetic Monday share is a little lower (14.96%) | A Berlin study found weekends busier (median 290 vs 245 visits/day), so the pattern is not universal (Frontiers Public Health 2022, https://www.frontiersin.org/articles/10.3389/fpubh.2022.711235/full) |
| Holiday effect | Urgent-care centres saw volume about 16% lower on July 4 and about 55% lower on Christmas Day. | JUCM 2023 (https://www.jucm.com/wp-content/uploads/2023/11/2023-18321-23-Practice-Management.pdf) | Direction agrees; synthetic effect is smaller (-12% in the generator rule) | Urgent care, not ED; no ED holiday benchmark found |
| Shortage / backorder frequency | A large distributor said essential-item fill rates were historically 96-98% and were in the high 80s during the 2022 supply crisis; hospitals once dealt with 50-100 back-ordered items per day. | Becker's Hospital Review, May 2022 (https://www.beckershospitalreview.com/supply-chain/lifesaving-medical-supply-shortages-persist-amid-supply-chain-disruptions) | **Does not agree**: the synthetic fill-rate proxy (99.88%) is above even the normal-time distributor range | Different metrics (distributor-to-hospital fill rate vs item-day backorder); trade-press report |
| Use rates per arrival, lead times, case packs, surge size | No independent benchmark found | n/a | Unvalidated | These are manually constructed assumptions (`source_status = manually_constructed_assumption`) |

## Conclusion

* Supported (qualitatively): arrival scale and the Monday-high, weekend-low pattern.
* Not supported: use rates, lead times, surge magnitude, holiday size for an ED.
* Contradicted or at risk: **shortage frequency is probably understated**, which matters because the shortage model has only 24 positive test rows and is sensitive to generator assumptions (TEST-30).
* TEST-17 therefore stays *Partial pass*. Closing it needs expert review or an authorized hospital extract.
