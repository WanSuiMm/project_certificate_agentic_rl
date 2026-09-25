# GPT handoff: Oracle Credit P2 follow-up

- Review base: `290a9d22ef50c89963fcd928add22e8cfbc9623f`
- Evidence head: `bc1c7f9af89b546ba3e12f6684d1c797a0110add`
- This handoff is metadata-only. Review the evidence-head delta first.

## Read first

1. [`ORACLE_P2_POLICY_DYNAMICS_20260925.md`](ORACLE_P2_POLICY_DYNAMICS_20260925.md):
   motivation from the supplied GPT note, corrected first-hit count, new
   depth-one q analysis, and limits.
2. [`evidence/oracle_credit_p2_q_v01/analysis.json`](evidence/oracle_credit_p2_q_v01/analysis.json):
   reproducible matched-support task metrics. The 384-row source-free
   [`observations.jsonl`](evidence/oracle_credit_p2_q_v01/observations.jsonl)
   is secondary; do not open it first.
3. [`ORACLE_CREDIT_RESULTS_20260925.md`](ORACLE_CREDIT_RESULTS_20260925.md):
   original K=4 fixed-P8 result, unchanged from the review base.

## What changed

The frozen Oracle Credit 6 × 16 × 4 trajectories were **not regenerated**.
Their first-hit selected-public-test outcome is 53/384, versus 14/384 passing
at P8. The supplied note had quoted 57/384; the published analysis corrects
that arithmetic against the original step journal. Frozen-bank q was measured
offline on the existing P2 sources: 373/384 numeric observations, with 320
new unique-source observer calls and 43 distinct sources reused from P1.

On matched P1-unsolved, valid-body candidates with tied P1 public-test scores,
mean P2 q and mean q drift show **mixed** descriptive direction against the
noisy first-hit continuation value: two of three informative tasks improve
over raw q in some comparisons, one worsens. All three tasks are from one
repository. P2 is drawn from the same K=4 continuations as the target, so the
result is in-sample and cannot establish predictive generalization. No trained
PRM, new policy rollout, or online RL result was added.

The new code is [`scripts/grade_swesmith_oracle_p2_q.py`](scripts/grade_swesmith_oracle_p2_q.py)
for frozen-bank scoring and [`scripts/summarize_swesmith_oracle_p2_q.py`](scripts/summarize_swesmith_oracle_p2_q.py)
for a source-free first-hit recount and comparison. The full source run and
private q bank were not uploaded. Existing theory synthesis and original
September 21 handoff are unchanged; see
[`THEORY_AND_HANDOFF_SUMMARY_20260925.md`](THEORY_AND_HANDOFF_SUMMARY_20260925.md)
only if that earlier context is needed.

## Reviewer questions

1. Does the first-hit recount correctly separate selected-test first success
   from P8 survival and hidden correctness?
2. Does the matched-support table justify only a mixed descriptive finding,
   especially given K=4, dependent candidate pairs and same-continuation P2?
3. Is the local semantic-drift hypothesis stated as motivation rather than an
   established cheap advantage or PRM proxy?
