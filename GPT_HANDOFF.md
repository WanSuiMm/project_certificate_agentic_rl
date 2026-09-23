# GPT handoff: exact-25 replay audit

- Review base: `340b492ca63bf89864ff297a78bc048cc363c9be`
- Evidence head: `473ad6828cd25960d1168a5cc4ff19d0d5e97a34`
- This handoff is a metadata-only follow-up; the evidence head above is stable.

## Read first

1. [`EXACT25_RESULTS_AND_NEXT_v01.md`](EXACT25_RESULTS_AND_NEXT_v01.md):
   result, invalid-case diagnoses, and next decision.
2. [`RESULTS.md`](RESULTS.md): aggregate table and prior Moto context.
3. [`GPT_CONTEXT.md`](GPT_CONTEXT.md): claim boundary and code map.

Only open [`aggregate_summary.json`](runs/exact25_intermediate_v01/aggregate_summary.json)
to inspect individual curves; its 1,591 lines encode 184 states, not additional
prose. Do not start with raw trajectories, logs, or image assets.

## Decision-relevant change

The exact-bound batch has 25/25 task summaries but only 18/25 strict endpoint
gates. Across 184 states, 119 have valid `(F,R)` observations. Two valid
negative transitions exist (Moto and SQLFluff). Seven invalid endpoints are
retained and diagnosed as runner/environment issues. A naive 23/25 endpoint
agreement is **not** a pass claim.

No live-agent continuation, certificate predictor comparison, or online RL was
run. Next: repair test-runner semantics and dependencies in a new run version,
then preregister matched-budget live-agent continuations and a held-out
certificate-vs-public-test prediction comparison. Stop before RL if that gate
fails. The replay/aggregation code and tests are included in this repository.

## Reviewer questions

1. Are the seven invalid endpoints correctly kept outside valid `(F,R)` analysis
   without silently excluding them from the denominator?
2. Is the proposed live-agent continuation comparison sufficient to test
   future-value information without suffix leakage before RL?
