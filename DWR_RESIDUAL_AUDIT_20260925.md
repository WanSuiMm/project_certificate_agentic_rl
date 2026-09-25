# Local DWR residual audit (exploratory)

**Result:** the frozen 256-probe `q` can be decomposed and reproduced locally, but this small pilot does **not** show a robust predictive gain from grouped residuals over scalar `q`. No coding policy was trained and no new trajectory was generated; a fixed ridge predictor was fit only for this offline comparison.

## Question and frozen comparison

Can a coarse vector of program-behavior residuals rank future repair outcomes better than the scalar mean `q(P2)`? We split the existing 256 probe positions into **eight contiguous groups of 32** (primary) or 16 groups of 16 (resolution check), without using outcomes to choose groups. Each component is the reference-agreement fraction in that group. Their uniform mean is exactly the existing `q`; the tested extra flexibility is a goal/outcome-weighted combination of components.

For each saved P2 branch, the outcome is **any selected-public-test pass at P3–P8**, not merely the forced P8 endpoint. The comparison is within a frozen P1 candidate's four continuations. We exclude P1 and P2 states already passing, P2 body-invalid branches, missing `q`, and unavailable outcomes. Positive and negative branches within the same candidate form evaluable pairs; proxy ties receive 0.5. A fixed `Ridge(alpha=10)` learns group weights with source-hash exclusion under leave-one-P1-candidate-out or leave-one-task-out validation. No hyperparameter or bin boundary was selected from the result.

## Local reproduction and support

The isolated WSL observer re-executed saved source against the original fixed input/reference bank. The supporting `string2string` checkout was pinned to `c4a72f59aafe8db42c4015709078064535dc4191`. `bubblewrap` exposed only the task package, observer script, explicit Python dependencies, system libraries, and a temporary directory; network and host mounts were absent. Two package initializers were made lazy *inside the temporary mirror only* to avoid unrelated eager `torch` imports. The task target modules and 256 probes were unchanged. **All 264 distinct locally measured source states exactly matched their frozen scalar `q`.** This checks aggregate parity, not bit-for-bit environment equivalence for every observation.

The analytic support is **195 valid unsolved P2 branches**, with **26 P3–P8 first hits** (11 P8 endpoint passes). Only **15 mixed-outcome P1 candidates**, 41 positive–negative branch pairs, across **three informative tasks** can test ranking. All three tasks are from `string2string`; pairs and branches are not independent task units.

| Score | Candidate-macro pair accuracy | Task-macro pair accuracy |
|---|---:|---:|
| Public-test fraction `p(P2)` | 55.0% | 54.2% |
| Scalar `q(P2)` | 58.6% | 56.2% |
| 8-bin weighted residual, leave one candidate out | 49.7% | 50.7% |
| 16-bin weighted residual, leave one candidate out | 49.7% | 50.7% |
| 8-bin weighted residual, leave one task out | 60.3% | 57.6% |
| 16-bin weighted residual, leave one task out | 58.6% | 49.3% |

The 8-bin task holdout is slightly above scalar `q`, but the candidate holdout is below both `q` and public tests; the 16-bin task-macro result also deteriorates. With only three informative tasks, this is **not evidence of a reliable DWR gain**. It also does not falsify richer, semantically specified obligations or state-conditioned adjoints: these bins are a deliberately cheap representation, and the target is a noisy selected-test first hit under one frozen policy. A larger or more flexible model would be especially easy to overfit here.

## Reproduction and provenance

- Collector: [`scripts/audit_dwr_residual_local.py`](scripts/audit_dwr_residual_local.py). It requires WSL Python, `bubblewrap`, the pinned public checkout and local `numpy`/`joblib`/`tqdm` packages. It fails closed on any frozen-`q` mismatch.
- Source-free reproduction from a GitHub clone: `python scripts/analyze_dwr_residual_local.py --from-public-panel` from the repository root, with `numpy` and `scikit-learn` installed. This regenerated `analysis.json` byte-for-byte locally.
- Published artifacts: [`vectors.jsonl`](runs/dwr_residual_audit_v01/vectors.jsonl) contains 264 per-probe bit vectors, [`panel_public.jsonl`](runs/dwr_residual_audit_v01/panel_public.jsonl) contains the 195 source-free analysis branches, and [`analysis.json`](runs/dwr_residual_audit_v01/analysis.json) contains the aggregate counts. Full candidate source remains local and is not uploaded; re-executing the collector requires that original local Oracle run and frozen bank.

**Decision:** do not promote grouped residuals to a process reward or launch RL from this pilot. The current evidence supports local exact remeasurement and a negative/inconclusive 8/16-bin screen only.
