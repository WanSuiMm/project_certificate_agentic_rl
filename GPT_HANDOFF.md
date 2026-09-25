# GPT handoff: completed Oracle Credit pilot

- Review base: `375e25a` (previous GitHub handoff)
- Evidence head: `544f5f76d2b6a08648895cd32f8991ecf1e2996c`
- This handoff is metadata-only; review the evidence-head delta first.

## Read first

1. [`ORACLE_CREDIT_RESULTS_20260925.md`](ORACLE_CREDIT_RESULTS_20260925.md):
   canonical completed result and claim limits.
2. [`evidence/oracle_credit_6x16x4_v01/summary.json`](evidence/oracle_credit_6x16x4_v01/summary.json):
   all candidate p, q, Monte Carlo Q, uncertainty and per-task comparisons.
3. [`ORACLE_CREDIT_BENCHMARK_v01.md`](ORACLE_CREDIT_BENCHMARK_v01.md):
   frozen question, protocol, outcome definition and recovery semantics.
4. [`scripts/run_swesmith_oracle_credit.py`](scripts/run_swesmith_oracle_credit.py)
   and [`scripts/summarize_swesmith_oracle_credit.py`](scripts/summarize_swesmith_oracle_credit.py):
   generation/feedback loop and analysis implementation.

## Decision-relevant delta

The frozen-policy 6-task × 16-candidate × 4-continuation pilot is now complete:
384/384 P8 outcomes, 2,688/2,688 scored edits, 14 selected-public-test
successes. The initial interrupted journal was preserved and imported into a
new buffered run, which completed without discarding its frozen actions. The
public evidence includes source-free per-state and terminal rows plus raw-file
hashes; private source text, issue text, completions, test feedback, and launch
receipts were not uploaded.

The outcome is **INCONCLUSIVE**, not a q win: there are no within-task Q pairs
with disjoint 95% Wilson intervals at K=4. Descriptively, on the three tasks
with nonconstant empirical Q (all one repository), matched-support q pair
accuracy is 36.6% versus p 49.5%. Two task directions are unfavorable to q,
but the noisy Q and dependent candidate pairs do not support a decisive
negative claim. Three candidates already solved at P1, yet only 1/12 forced
P8 continuations remained solved; the benchmark measures value under this
specific seven-more-edit frozen policy, not intrinsic patch quality.

Unchanged: the one-edit census shows extra q resolution, not future-value
alignment; the earlier fixed-data LoRA result is not an on-policy long-horizon
RL comparison; the proposed matched Test/Semantic GRPO intervention remains
unrun. No PRM benchmark or critic comparison was run.

Local verification: 75 tests passed; the staged files and public export passed
row-count, link, whitespace, and sensitive-pattern checks. The downloaded raw
summary/state/outcome SHA-256 values match the server run; deployed runner and
scorer hashes match the committed source.

## Reviewer questions

1. Is the K=4 Wilson non-overlap rule too conservative to serve as a useful
   pilot screen, while still correctly preventing a positive claim here?
2. Do the low or negative descriptive q/Q associations on the three informative
   tasks suggest a protocol-specific mismatch between P1 agreement and forced
   P8 continuation value, beyond Monte Carlo noise?
3. What is the smallest future-value test that would distinguish these
   explanations without claiming an RL result from the current pilot?
