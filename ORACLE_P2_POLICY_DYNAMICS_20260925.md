# Oracle Credit P2: policy-conditioned semantic drift

This note organizes the user-supplied interpretation of the completed Oracle
Credit pilot and reports the subsequent **offline** measurement of `q(P2)`.
The proposal is a hypothesis, not a validated cheap process reward model.

## Why measure P2?

The frozen `q(P)` is agreement between a candidate program and the clean
reference on the task's fixed 256-input bank. It measures **present behavior**.
Continuation value also depends on how the frozen agent policy edits that
program. A high-`q` corner-case bug may be difficult for the policy, while a
low-`q` systematic bug may be easy to repair. The proposed depth-one signals
for a frozen P1 candidate are

\[
S_q(P_1)=\frac14\sum_{k=1}^4 q(P_2^{(k)}),\qquad
D_q(P_1)=S_q(P_1)-q(P_1).
\]

These use the **existing four** closed-loop continuations. No new action was
generated, and the policy never observed `q`. The comparison target is
`Q_hit(P1)`: the fraction of those continuations that pass the selected public
tests at least once from P1 through P8. This is a first-hit recount of the
recorded states, not a hidden-correctness oracle or a new Monte Carlo sample.

## Correction to the motivating note

The supplied note quoted **57/384** first-hit trajectories and P2/P3 counts
of 15/6. Recounting the frozen public state file gives **53/384**:

| First passing state | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trajectories | 12 | 13 | 4 | 7 | 3 | 4 | 3 | 7 |

Only **14/384** remain passing at P8. The fixed-horizon endpoint therefore
mislabels many trajectories that already passed and were then forced to keep
editing. The `Q_hit` recount repairs that *target definition* without rerunning
the actor, but passing the selected tests is still not general correctness.

## P2 measurement and observed signal

The original complete run contains 384 P2 states. After source-hash
deduplication there are 363 distinct P2 sources: 320 received new frozen-bank
observer calls and 43 are identical to a P1 source with a saved `q` (59 of the
384 observations). A P1 control was remeasured in every task image before
grading that task. **373/384** P2 observations have a numeric `q`; 11 are
observer-invalid or unavailable. Among the 363 observations with both P1 and
P2 `q`, 40 increased, 117 decreased, and 206 tied exactly. These are state
descriptions, not action-quality labels.

The table compares within-task candidate pairs with unequal empirical
`Q_hit`, **equal P1 public-test fraction**, and P1 unsolved with a valid body.
All four P2 `q` values and P1 `q` must be present for a candidate. Proxy ties
score 0.5; each cell shares the same candidate support within its task.

| Task suffix | Eligible pairs | `q(P1)` | mean `q(P2)` | mean `q(P2)-q(P1)` |
|---|---:|---:|---:|---:|
| `jkircbwq` | 18 | 63.9% | 58.3% | 47.2% |
| `js80kw75` | 41 | 37.8% | 64.6% | 84.1% |
| `h58dkipe` | 13 | 30.8% | 34.6% | 57.7% |

The other three tasks have **no unequal candidate `Q_hit` pair** on this
support. All three informative tasks come from `string2string`. The depth-one
signal improves some descriptive comparisons, but not consistently: on
`jkircbwq`, mean P2 `q` and drift are worse than P1 `q`. Pair counts are not
independent statistical units, and K=4 makes each empirical `Q_hit` coarse.
Moreover, P2 is part of the *same* continuations used to calculate `Q_hit`,
including 13 first hits at P2. This is an in-sample lookahead diagnostic, not
independent validation of prediction or a causal credit result. It does not
justify a KEEP verdict, PRM claim, or RL-improvement claim.

The original note also correctly flagged that nonoverlapping 95% Wilson
intervals are a nearly impossible discrimination gate at K=4; zero eligible
confident pairs is not evidence that candidate values are identical. Its
numerical-analysis analogy remains a useful **motivation**: program residual
and policy transition dynamics are distinct. This pilot does not yet establish
that their combination estimates long-horizon value reliably.

## Reproduce and inspect

Run from the repository root:

```bash
python scripts/summarize_swesmith_oracle_p2_q.py \
  --oracle-public-dir evidence/oracle_credit_6x16x4_v01 \
  --p2-observations evidence/oracle_credit_p2_q_v01/observations.jsonl \
  --output evidence/oracle_credit_p2_q_v01/analysis.json
```

Start with [`analysis.json`](evidence/oracle_credit_p2_q_v01/analysis.json)
and the source-free [`observations.jsonl`](evidence/oracle_credit_p2_q_v01/observations.jsonl).
[`run.json`](evidence/oracle_credit_p2_q_v01/run.json) pins input hashes and
completion counts. [`measurements.jsonl`](evidence/oracle_credit_p2_q_v01/measurements.jsonl)
is the deduplicated observer journal. Full source and model completions remain
local and are intentionally excluded from GitHub. Re-running the observer
itself additionally needs the pinned task metadata, frozen 256-input q bank,
official SWE images and Modal access; those inputs are not redistributed here.
The source-free recount and comparison above run from the published files.
