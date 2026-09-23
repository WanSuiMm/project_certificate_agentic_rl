# GPT handoff: eight-step semantic GRPO code, no new run

- Review base: `885275c3f7fabead88a20fc523b6cfa7259e5ecb`
- Evidence head: `a3b5a52` (experiment code, protocol, and tests)
- This handoff is metadata-only. Review the evidence-head delta first.

## Read first

1. [`LONG_HORIZON_GRPO_PROTOCOL_20260924.md`](LONG_HORIZON_GRPO_PROTOCOL_20260924.md): three experiments, reward definitions, and claim limits.
2. [`CLOSED_LOOP_TRAJECTORY_PROTOCOL_20260924.md`](CLOSED_LOOP_TRAJECTORY_PROTOCOL_20260924.md): public-feedback rollout and offline q ordering.
3. [`RESULTS.md`](RESULTS.md): canonical results, including the unchanged negative/offline evidence.
4. [`scripts/train_swesmith_long_horizon_grpo.py`](scripts/train_swesmith_long_horizon_grpo.py): on-policy Test and Semantic arms; [`scripts/summarize_swesmith_trajectory_grpo_signal.py`](scripts/summarize_swesmith_trajectory_grpo_signal.py): frozen-policy census summary.

## Decision-relevant delta

The earlier open-loop/no-feedback capture was not silently reused. A new eight-edit closed-loop pipeline records public-test feedback after each edit, keeps q hidden from the policy, then computes q for all P0–P8 states offline. It uses a persistent sandbox per task, transfers the 256-case q bank once, and deduplicates identical source states.

The primary intervention now has two matched on-policy whole-trajectory GRPO arms: `Y8 + 0.5 p8` and `Y8 + 0.5 q8`. Each task group has 16 fresh trajectories, eight edits each, with one group-relative advantage per trajectory. An opt-in step-wise credit extension exists but is not the primary experiment. `Y8` denotes selected public-test resolution, not an independent hidden grader.

The server was off: **none of these three new experiments has run**. No new scientific result or timing claim follows from this commit. The previous 54 open-loop trajectories and fixed-P1 LoRA comparison remain separate, unchanged diagnostics. Local verification: `python -m pytest -q tests` passed 61 tests; staged diff and secret scans passed.

## Reviewer questions

1. Does the frozen-policy eight-step census show Semantic rescue of groups tied under Test reward?
2. On matched fresh policy rollouts, does Semantic improve held-out public-test solve@8 versus Test at equal environment steps?
3. Only if the primary comparison is informative, does step-wise credit add benefit beyond whole-trajectory GRPO?
