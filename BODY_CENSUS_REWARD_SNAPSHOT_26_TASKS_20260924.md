# One-edit reward signal: provisional 26-task snapshot

This is a frozen **partial snapshot**, not the 28-task census endpoint or an RL
training result. At approximately 2026-09-23 16:45 UTC, 419/448 rows had been
recorded; the first 26 frozen tasks had all 16 candidates each (416 candidates).
The remaining three rows belonged to the next incomplete task and are excluded
below. The raw snapshot is retained locally, not uploaded. Its SHA-256 is
`f0eb9ebe5f3953205da91dce129ea567619783f46c7676e12cbe52b47b71d9c9`;
the selection SHA-256 is
`7b8c94a6f81d48be6ac2c710d2edd992712a80318a8ce523e868e3bdf1846080`.
Regenerate the aggregates with `scripts/summarize_swesmith_reward_census.py`.

All 16 candidates per task came from the same frozen 1.5B base policy. Each
candidate was scored with terminal success `Y`, public-test fraction `p`, and
the fixed reference-agreement proxy `q`. The compared rewards are exactly
`R_test = Y + 0.5p` and `R_semantic = Y + 0.5q`. The table restricts reward
comparisons to candidates with executable, valid public-test observations;
invalid candidates are not treated as informative zero-reward distinctions.

| Measure on 26 complete tasks | Snapshot |
| --- | ---: |
| Recorded candidates | 416 |
| Executable candidates | 328/416 |
| Solved candidates | 43/328, across 9 tasks |
| Tasks with varying terminal `Y` | 9/26 |
| Tasks with varying `R_test` among executable candidates | 11/26 |
| Tasks with varying `R_semantic` among executable candidates | 25/26 |
| Tasks where `R_test` is tied but `R_semantic` varies | 14/26 |
| Tasks with at least one unresolved test-reward tie split by semantic reward | 23/26 |
| Unresolved, equal-`R_test` within-task candidate pairs | 1,370 |
| Such pairs with unequal `R_semantic` | 607/1,370 |

The task—not the 1,370 dependent pairs—is the meaningful unit for judging
coverage. The signal is strong **as an information/credit-availability result**:
semantic reward frequently distinguishes unresolved candidates that terminal
and public-test reward cannot distinguish. It does not establish that higher
`q` is a better action, predicts future repair, or improves LoRA training.
Indeed, two executable unresolved candidates had `q=1` on the finite probe
bank, so `q` is not a correctness oracle. The q-first selection is deliberately
enriched for informative tasks and is not a prevalence estimate for SWE-smith.
Moreover, 43 one-edit candidates already solve their task, so terminal reward
is not uniformly sparse in this restricted one-edit setting.

No optimizer update or held-out policy evaluation occurred in this census.
The separate eight-step observational continuation was still running at this
snapshot, and its all-state offline `q(P0),...,q(P8)` grading had not started.
Consequently this snapshot supports **extra reward resolution**, not the
project's long-horizon temporal-credit or RL-improvement claim. Those remain
open until full trajectories and an actual matched Test-vs-Semantic LoRA
intervention are evaluated.
