# GPT handoff: theory, conditional story, and related work

- Review base: `06c65cbf43d4517494ccf4fca6bc03e9e7747e1f`
- Evidence head: `519b01e5d64fcbb9d5f5f4dc061082e344d2d1e3`
- This handoff is metadata-only; review the evidence-head delta first.

## Read first

1. [`THEORY_STORY_AND_RELATED_WORK_20260925.md`](THEORY_STORY_AND_RELATED_WORK_20260925.md):
   synthesis of the three supplied notes. It distinguishes the theory recheck,
   a *conditional* paper storyline, and verified related-work positioning.
2. [`THEORY_AND_HANDOFF_SUMMARY_20260925.md`](THEORY_AND_HANDOFF_SUMMARY_20260925.md):
   original theory/adapter provenance and numerical-check limits.
3. [`ORACLE_P2_POLICY_DYNAMICS_20260925.md`](ORACLE_P2_POLICY_DYNAMICS_20260925.md):
   latest empirical result; unchanged in this delta. The source-free aggregate
   is [`analysis.json`](evidence/oracle_credit_p2_q_v01/analysis.json).

## Decision-relevant delta

This update adds **no experiment, model training, code change, or improved
result**. It makes three supplied discussion notes available in a concise
GitHub reading path:

- The independent theory recheck separates current program error from
  policy-conditioned value. Correct terminal potential settlement preserves
  the original objective, while matched `Phi+W` and `V` critics have identical
  TD/GAE residuals. `Delta q` is not generally action advantage.
- The proposed strong storyline requires a chain of still-unproven results:
  semantic resolution, held-out future-value relevance, a better matched
  critic, and online solve-rate gains under the **same** task reward. Its paper
  narrative is not a result or claim of novelty.
- The literature map points to MC step-importance, AgentPRM, VPR,
  SWE-Shepherd and RUDDER via checked primary sources. It corrects the
  supplied note's VPR task examples and does not assert an exhaustive novelty
  search or a completed PRM quality–cost comparison.

The prior Oracle Credit result remains noisy and mixed: 53/384 first-hit
selected-test continuations, 373/384 numeric P2 q observations, and no
independent validation that static q or depth-one drift estimates long-horizon
value. No online RL result exists.

## Reviewer questions

1. Are the algebraic identities, finite-world checks, empirical SWE findings,
   and aspirational critic experiment clearly separated?
2. Does the related-work positioning cite comparable methods without claiming
   the current pilot has already matched their training or evaluation?
3. Is the strongest paper story explicitly conditional on held-out value and
   same-reward online RL evidence?
