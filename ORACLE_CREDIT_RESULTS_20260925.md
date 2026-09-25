# Oracle Credit Benchmark v01: completed pilot

The frozen-policy pilot finished all 6 selected SWE-smith tasks, 16 frozen P1
candidates per task, and 4 closed-loop P1-to-P8 continuations per candidate:
**384 endpoints and 2,688 scored edits**. Fourteen endpoints passed the selected
public tests. This is **not** LoRA/RL training, and the P8 pass criterion is not
an independent hidden-correctness grader.

## Primary question and result

Does P1 reference-behavior agreement `q` order the future success probability
of frozen-policy continuations better than the agent-visible public-test pass
fraction `p`, especially when `p` ties? **Inconclusive.** No within-task pair
of empirical continuation values had nonoverlapping two-sided 95% Wilson
intervals. With only K=4 continuations, observed candidate values are multiples
of 0.25. The predeclared confident-pair comparison therefore has zero eligible
pairs and cannot establish either a win or a loss for `q`.

| Task suffix | P8 successes / 64 | P1 candidates with any P8 success | Descriptive `q` pair accuracy on non-tied empirical Q pairs |
|---|---:|---:|---:|
| `gch4emzr` | 0 | 0 | undefined |
| `jkircbwq` | 7 | 6 | 53.8% |
| `fhdnra9k` | 0 | 0 | undefined |
| `js80kw75` | 5 | 5 | 27.3% |
| `h58dkipe` | 2 | 1 | 28.6% |
| `atmgadcu` | 0 | 0 | undefined |

The three tasks with a nonconstant empirical Q all come from **one repository**
(`string2string`). On matched q-valid candidates, the equal-task-weighted
descriptive pair accuracies are `q=36.6%` and `p=49.5%`; proxy ties score
one-half. The public-test-tied, P1-unsolved subset gives q accuracies of
66.7%, 27.3%, and 28.6% in those three tasks. These pairs share candidates,
are not independent trials, and rest on noisy K=4 values. The descriptive
direction is unfavorable to q in two tasks but is **not** a statistically
resolved negative claim.

The fixed-horizon target itself matters. Three P1 candidates already passed
the selected public tests, yet only 1 of their 12 forced continuations still
passed at P8. Thus this benchmark estimates value under this particular frozen
policy and seven-more-edits protocol, not intrinsic patch quality or optimal
repairability. High P1 q can be erased by subsequent edits; low P1 q can still
be repaired. The earlier one-edit census established extra *resolution* of q,
not this future-value alignment.

## Evidence and boundaries

Read [`evidence/oracle_credit_6x16x4_v01/summary.json`](evidence/oracle_credit_6x16x4_v01/summary.json)
for all candidate p/q/Q values, per-task pair metrics, and completeness checks.
[`outcomes_public.jsonl`](evidence/oracle_credit_6x16x4_v01/outcomes_public.jsonl)
contains all 384 terminal outcomes; [`states_public.jsonl`](evidence/oracle_credit_6x16x4_v01/states_public.jsonl)
contains all 2,688 scored edit states without source text or completions.
[`manifest.json`](evidence/oracle_credit_6x16x4_v01/manifest.json) pins the
input/raw artifact hashes, model revision, counts, and export scope. The full
163 MB raw run is retained locally; source, issue text, action completions,
test feedback text, logs, and machine-specific launch receipts are intentionally
not published. The public export can be reproduced with
`scripts/export_swesmith_oracle_public.py` against that raw run.

The completed run resumed from a preserved interrupted checkpoint (1,600 edits,
192 endpoints) into a new buffered run. Its importer checked pinned inputs and
journal provenance, then completed the remaining edits without resampling prior
actions. See [`ORACLE_CREDIT_BENCHMARK_v01.md`](ORACLE_CREDIT_BENCHMARK_v01.md)
for the frozen protocol and exact code entry points. No PRM comparison or
on-policy RL intervention was run. Do not turn this pilot into a KEEP/KILL
verdict or a broad multi-repository estimate.
