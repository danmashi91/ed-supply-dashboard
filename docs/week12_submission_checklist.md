# Weeks 12-13 Submission Checklist

## Completed and ready in the repository

- [x] Completed validation matrix using the required Excel template
- [x] Organized validation evidence index
- [x] Reproducible generation and model-training code
- [x] Fresh-environment reproducibility record
- [x] Updated Streamlit application from Week 11
- [x] Two-to-three-page validation memo
- [x] Synthetic-data provenance, dictionary, assumptions and validation results
- [x] Model evaluation, baseline comparison, leakage checks and error analysis
- [x] User-testing protocol and results for two outside substitute participants
- [x] Typical end-to-end walkthrough
- [x] Challenging end-to-end walkthrough
- [x] Previous Week 10 submission preserved as a tag and snapshot branch
- [x] Updated group-contribution summary

## Still required before Canvas submission

- [ ] Record and upload the seven-minute demonstration video
- [ ] Record and upload the additional twelve-minute matrix walkthrough video
- [ ] Add both accessible video links to `README.md`
- [ ] Confirm that the live Streamlit application opens from a logged-out or private browser window
- [ ] Confirm that the instructor and both group members can access the repository and video links
- [ ] Submit the GitHub repository link in Canvas

## Closed in the 15 Nov revision

- [x] Standalone decision-support requirements (`docs/decision_support_requirements.md`)
- [x] Cross-page KPI and filter reconciliation (TEST-15, 90 checks) and Scenario Comparison / Recommended-actions reconciliation (TEST-21, 22)
- [x] Normal, boundary and invalid cases for reports and what-if inputs (TEST-06, 23, 25)
- [x] Saved generated-summary samples and reconciliation (TEST-14, 24)
- [x] Invalid-input run evidence and screenshot (TEST-06)
- [x] Baseline comparisons (TEST-27, 28) and sensitivity to generation assumptions (TEST-29, 30)
- [x] Independent benchmark comparison where sources exist (TEST-17, still partial)
- [x] Reproducible test scripts and instructions (`validation/`)

## Still open after the revision

- [ ] TEST-31: outside-user task session (`docs/user_testing_round2_protocol.md`)
- [ ] TEST-30 failed: add the shortage-model sensitivity caution to the app or accept it as a documented limitation
- [ ] Group members review the revised matrix and fill in the owner column for tests run by the automated suite
- [ ] Commit and push the revision, then confirm the live app still loads

## Correctly retained evidence gaps

These gaps should not be relabelled as passes without new evidence:

- Independent empirical benchmarking of all major synthetic variables remains incomplete.
- Existing user testing used substitutes rather than an ED inventory manager and did not fully test explanation interpretation.
- Operational controls for safety stock, open orders, lead-time uncertainty, cost, storage and expiration are not implemented.
