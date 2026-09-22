# Moto intermediate replay protocol v0.1

**Status:** frozen before inspecting Moto intermediate test curves

## Scope

Run the five exact-bound selected Moto trajectories (two declared resolved,
three declared unresolved) on the prepared PRoot execution substrate. This is an
engineering and first-signal slice, not a repository-independent estimate.

## State sequence

For each task, apply the pinned official task mutation to the clean image and
call the resulting initialized workspace `P0`. Replay every recorded filesystem
mutation in trajectory order to obtain `P1, ..., PK`. Retain auxiliary file
creates as mutation boundaries; tag them separately from source edits rather
than silently dropping them.

At every `Pk`, record:

- changed and untracked paths plus content hashes;
- the exact preceding tool call;
- raw FAIL_TO_PASS and PASS_TO_PASS test output;
- `F_k`: number of FAIL_TO_PASS tests currently failing;
- `R_k`: number of PASS_TO_PASS tests currently failing;
- timeout, collection error, or environment error as `INVALID`, never as an
  ordinary test failure count.

## Frozen transition labels

For two valid adjacent states:

- `positive`: `F` and `R` do not increase, and at least one strictly decreases;
- `negative`: `F` and `R` do not decrease, and at least one strictly increases;
- `mixed`: one decreases while the other increases;
- `neutral`: both are unchanged.

No scalar weighting of `F` and `R` is allowed in this slice.

## Per-task execution gate

`PRoot_ENDPOINT_REPLAY_PASS` requires:

1. the clean image state is readable and the selected tests execute;
2. the official task mutation reproduces at least one FAIL_TO_PASS failure;
3. recorded mutations replay sequentially with fail-closed semantics;
4. every intermediate workspace and test state is saved;
5. the final endpoint agrees with the declared outcome, or any discrepancy is
   retained and assigned an explicit data, tool-semantics, or runtime cause.

No mismatching task may be silently removed after its outcome is observed.

## Pre-run exposure audit

Before replay, report `N_undo` and the count of trajectories with suspicious
shell-side mutation commands in the 25-task exact stratum. Implement additional
semantics only when exercised. Any unsupported exercised action makes the
affected task `INVALID` until support is added; it does not authorize post-hoc
exclusion.

## Interpretation

Moto can demonstrate that intermediate test-vector structure is observable and
can supply motivating positive, negative, mixed, or non-monotone examples. It
cannot establish cross-repository prevalence, certificate value-prediction gain,
or an online-RL result.
