# GPT handoff: Moto intermediate audit v0.1

- Review base: `98c12d775d9110aebc524729b052b11cb92dd0f1`
- Evidence head: `c1e30ea678237c264d3c07388849570332e35a1f`
- Scope: first five-task exact-bound Moto intermediate replay

## Read only these first

1. [`RESULTS.md`](RESULTS.md), especially “Moto intermediate audit v0.1”.
2. [`runs/moto_intermediate_v01/aggregate_summary.json`](runs/moto_intermediate_v01/aggregate_summary.json).
3. [`MOTO_INTERMEDIATE_PROTOCOL_v01.md`](MOTO_INTERMEDIATE_PROTOCOL_v01.md).
4. [`GPT_CONTEXT.md`](GPT_CONTEXT.md) for the current claim boundary.

Do not begin with the per-task summaries or replay-surface row file; they are
supporting evidence for the compact aggregate.

## What changed

- Replayed two resolved and three unresolved exact Moto trajectories from fresh
  image-derived root filesystems.
- Recorded FAIL_TO_PASS and PASS_TO_PASS outcomes after every confirmed editor
  mutation: 52 states, 47 successful mutations, 48 attempted mutations.
- Added tool-outcome filtering so a failed editor call does not create a fake
  state transition.
- Added the exact-25 undo/shell-mutation surface audit. Moto has no such replay
  complication; three non-Moto exact rows have suspicious shell-side mutations.

## Decision-relevant evidence

- Endpoint gate: 5/5 pass and `PRoot_ENDPOINT_REPLAY_PASS=true`.
- `pr_7144`: `(4,0) -> (4,12) -> (4,1)`, a valid negative excursion in an
  ultimately unresolved trajectory.
- `pr_6509`: `(1,0) -> (0,0) -> invalid x8 -> (0,0)`. The invalid region begins
  with a trajectory edit that imports an unavailable symbol and ends when a
  later edit repairs the module; it is not test-induced workspace drift.
- Three positive, one negative, 34 neutral, and nine invalid-adjacent
  transitions were observed. This is a one-repository signal slice, not a
  prevalence estimate.

## Unchanged claim boundary

- No certificate feature has been evaluated on these states yet.
- No critic comparison, held-out predictive gain, sample-efficiency result, or
  online-RL claim exists.
- The selected panel remains selection-conditioned and the 15 profile-fallback
  rows remain outside the primary exact stratum.

## Reviewer questions

1. Is the negative `pr_7144` excursion sufficient to justify expanding the same
   frozen phenotype audit across repositories before implementing a critic?
2. Should collection-invalid states be represented as a distinct categorical
   state or treated as censored observations outside `(F,R)`?
3. What is the cheapest certificate-derived feature to compute on the frozen
   state sequence without leaking future edits or terminal outcome?
