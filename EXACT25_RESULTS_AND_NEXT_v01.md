# Exact-25 trajectory audit: result and next decision

Status: completed batch, **partial replay qualification**, no agent/RL experiment.
The frozen exact-binding stratum contains 25 historical SWE-smith agent
trajectories. The runner applied each official task mutation to its image-derived
workspace, replayed recorded filesystem mutations, and tested every resulting
state under PRoot. This is offline replay of a fixed agent trace; neither a live
agent nor an RL policy was run. The compact, path-free result is
[`runs/exact25_intermediate_v01/aggregate_summary.json`](runs/exact25_intermediate_v01/aggregate_summary.json).

## Observed results

| Measure | Result | Boundary |
|---|---:|---|
| Task runs with a summary | 25/25 | Batch completion, not gate passage |
| Strict endpoint gate | 18/25 | Official patch applied; valid failing initial state; valid final state; declared outcome agrees |
| State observations | 184 | 119 valid `(F,R)`; 65 invalid test observations |
| Adjacent transitions | 159 | 13 positive; 85 neutral; 2 negative; 59 invalid-adjacent; 0 mixed |
| Naive endpoint agreement | 23/25 | Misleading: invalid unresolved endpoints can appear to agree |

The two valid negative transitions occur in Moto `pr_7144` and SQLFluff
`pr_5880`. The Moto excursion is `(4,0) -> (4,12) -> (4,1)`; the SQLFluff
excursion is `(1,0) -> (1,1) -> (1,0)`. These are observed test-vector
non-monotonicity, not evidence that any certificate feature predicts future
success. Moto `pr_6509` has eight invalid intermediate states caused by a
trajectory edit, but valid initial and final endpoints. The 25 trajectories were
selected for structured edits and are not an unbiased population sample.

Seven tasks fail the strict endpoint gate because test observations are invalid:

| Task | Current diagnosis | Classification |
|---|---|---|
| MONAI `pr_4745`, `pr_5423` | Image environment lacks importable `pytest` | Environment |
| Pydantic `pr_5483` | Missing `jsonschema` dependency | Environment |
| Hydra `pr_2543`, Pydicom `pr_1987`, Mypy `pr_12951` | Empty PASS_TO_PASS list makes the runner launch an unrestricted pytest suite; Pydicom times out | Runner semantics |
| Result `pr_135` | FAIL_TO_PASS collection cannot import `do`; empty PASS_TO_PASS also triggers unrestricted collection | Task/environment plus runner; unresolved |

These seven are retained in the denominator. The batch has no valid endpoint
disagreement; a failed test collection is not a valid `(F,R)` point. The current
result qualifies the replay substrate for a subset, but does **not** make the
exact-25 panel a clean 25-task scientific dataset. Docker parity is an external
robustness check, not a prerequisite for this PRoot audit.

## Next work, in order

1. Fix empty test-group semantics: skip a group with zero requested tests and
   mark it explicitly empty, never invoke unrestricted pytest. Add a regression
   test. Re-run only affected tasks in a new versioned output directory, keeping
   this v0.1 evidence unchanged.
2. Resolve image/test dependencies on the other invalid tasks with small
   clean-image and initialized-task smokes. Document whether each failure is
   environment support, task incompatibility, or a genuine trajectory state;
   do not turn collection errors into failures or exclude tasks post hoc.
3. Freeze a valid, repository-diverse pre-action checkpoint set and measure
   continuation outcomes with the **same agent/model, history, tool budget and
   test budget** across comparisons. Multiple stochastic continuations per
   checkpoint supply empirical future-return targets; split by task/repository
   and prevent suffix or outcome leakage. This is an online-agent *evaluation*
   design, not yet RL training.
4. Compare a cheap certificate-derived state feature against return-only and
   public-test `(F,R)` baselines at matched data and compute. Pre-register
   paired task-group Brier/MSE as the primary statistic and preserve the
   existing Gate-0 stop line (at least 32 informative version pairs and at least
   10 certified changes in each direction). If the certificate has no held-out
   incremental predictive value, stop before online RL.
5. Only after that gate passes, specify a bounded online RL intervention and
   its matched-compute control. No online RL result is claimed or launched here.

## Reproduction and code

- Replay engine: `scripts/run_moto_intermediate.py` and
  `scripts/replay_structured_edits.py`.
- Resumable 25-task orchestration: `scripts/run_exact25_remote_batch.py`.
- Public aggregate generator: `python scripts/summarize_exact25_intermediate.py
  --run-dir RUN_DIR --expected-tasks 25` (requires the retained 25 per-task
  `summary.json` files in `RUN_DIR/<instance_id>/`).
- Aggregate tests: `python -m unittest tests.test_exact25_summary -v`.

Image layers, raw trajectory rows, per-state logs/XML, and machine-specific
launch receipts remain outside the public review path. The aggregate carries
the per-task `(F,R)` curves and gate booleans, without server paths or credentials.
