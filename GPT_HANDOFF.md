# GPT handoff: local DWR residual audit

- Review base: `41674e9d2df5f576579d6072e70f881aad3a125c`
- Evidence head: `55cd820` (this handoff is a later metadata-only commit)
- Review the evidence-head delta, not the full repository history.

## Read first

1. [`DWR_RESIDUAL_AUDIT_20260925.md`](DWR_RESIDUAL_AUDIT_20260925.md): question, protocol, exact numbers and claim boundary.
2. [`analysis.json`](runs/dwr_residual_audit_v01/analysis.json): canonical aggregate; [`panel_public.jsonl`](runs/dwr_residual_audit_v01/panel_public.jsonl) is the source-free 195-branch reproduction input. Do not open per-state [`vectors.jsonl`](runs/dwr_residual_audit_v01/vectors.jsonl) first.
3. [`analyze_dwr_residual_local.py`](scripts/analyze_dwr_residual_local.py): run `python scripts/analyze_dwr_residual_local.py --from-public-panel` from the repository root with NumPy and scikit-learn. The local output matched `analysis.json` byte-for-byte. [`audit_dwr_residual_local.py`](scripts/audit_dwr_residual_local.py) is the isolated source remeasurement code; it requires the intentionally unpublished saved source run.

## Decision-relevant delta

The new local audit exactly reproduced the frozen scalar `q` for 264 distinct saved source states and decomposed their original 256-input reference-agreement vectors. Among 195 valid, unsolved P2 branches, 26 had a first selected-public-test pass at P3–P8. Only 15 P1 candidates across three tasks had both positive and negative branches for within-candidate ranking (41 pairs; not independent units).

Candidate-macro pair accuracy was 55.0% for public-test fraction, 58.6% for scalar `q(P2)`, and 49.7% for a fixed 8-bin weighted residual under leave-one-candidate-out validation. An 8-bin leave-one-task-out fit reached 60.3%, but the direction was not stable across the 16-bin check or validation choices. This is **negative/inconclusive for a reliable grouped-residual gain**, not evidence that DWR-style semantic obligations in general fail.

No coding policy was updated, no new trajectories were generated, and no RL or causal-credit claim changed. The prior Oracle Credit pilot, P2 drift result, and conditional theory story remain as previously reported. Full candidate source is local-only; the uploaded panel is sufficient to recompute the published statistics, not to re-execute the observer.

## Reviewer questions

1. Do the valid-unsolved-P2 filter, P3–P8 first-hit target, within-candidate comparisons and source-hash exclusion prevent the obvious leakage/mislabelling issues?
2. Does the report properly distinguish exact scalar-`q` parity from unproven Docker/whole-observation parity and distinguish three informative tasks from 41 correlated pairs?
3. Is the stopping conclusion appropriately limited to the fixed 8/16 contiguous-bin ridge screen rather than the broader DWR framework?
