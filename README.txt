APP BUILD FOLDER (data v1.1-2026-10-19)
=======================================

This folder is separate from the graded Week 7 submission. Nothing in the
parent folder was changed.

What is here
- code/generate_and_validate.py  Updated generator (v1.1). Run: python3 code/generate_and_validate.py
- data/                          Regenerated datasets (same file names and layout as Week 7)
- docs/app_specification.md      Spec to paste into a new AI chat to generate the Streamlit app
- docs/data_dictionary.csv, docs/pipeline_summary.json

What changed from Week 7 (v1.0) and why
The v1.0 data had only 8 high-risk item-days, all on 2024-01-01 (a lean-start
artifact), so a dashboard would have nothing to show. v1.1 adds:
1. Arrival surges: 6 multi-day episodes (+18-38%) and 8 one-day spikes (+45-70%). New column surge_flag.
2. Twice-weekly ordering (Mon/Thu, or any day risk >= 0.50). Order-up-to target
   covers lead time + safety stock + 4-day review period. New column order_placed_qty.
3. Supplier disruptions: 8% of orders arrive 2-6 days late; 5% are short-shipped
   (30-70% of units). New column supplier_event.
4. Starting stock = target stock.

Result: 223 high-risk item-days spread over all items and months, 19 backorder
days, fill rate 99.88%, 13/13 validation checks pass.

Next step: paste docs/app_specification.md into a new chat and ask it to
build the app; save the generated app.py here, then: streamlit run app.py
