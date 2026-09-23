# GPT handoff: provisional reward information, no RL outcome

- Review base: `cfbe828678b8a76f241b30a6eb857f6376f77225`
- Evidence head: `35f3b610a073b9d455a85d21b09e15e1d671066d`
- This handoff is metadata-only. Review the evidence-head delta; historical
  exact-25 replay and the stopped zero-gradient GRPO attempt are unchanged.

## Read first

1. [`BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md`](BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md):
   fixed partial reward snapshot and its claim boundary.
2. [`RESULTS.md`](RESULTS.md) and [`GPT_CONTEXT.md`](GPT_CONTEXT.md): formal
   status and code/evidence routing.
3. [`scripts/summarize_swesmith_reward_census.py`](scripts/summarize_swesmith_reward_census.py):
   exact task-level and within-task computations.

Raw completions, per-sample rows, source states, logs, and machine-specific
receipts are intentionally not uploaded. Do not open large JSONL first.

## Decision-relevant delta

At the fixed 26/28-task snapshot, 328/416 candidates were executable and 43
solved. On executable candidates, `Y+0.5p` varies on 11/26 tasks while
`Y+0.5q` varies on 25/26; 14/26 have tied test reward but varying semantic
reward. Among unresolved candidates, semantic reward splits 607/1,370
within-task test-reward-tied pairs, across 23/26 tasks. Task is the meaningful
coverage unit; pairs are dependent. Two unresolved executable candidates have
q=1 on the finite bank. This supports extra reward resolution, **not** useful
future credit, trained-policy improvement, or a result on the final 28-task
census.

The separate public-feedback-only eight-step observation remains in progress;
all-state offline q and matched Test-vs-Semantic LoRA training have not run.
The model is frozen during the census and trajectory observation. Local tests:
51 passed.

## Reviewer questions

1. Are the task-level reward-variation counts and executable-only denominator
   sufficient to support the narrow extra-resolution claim?
2. Could finite-bank q saturation (`q=1`, unresolved) invalidate the proposed
   use as shaping reward despite the observed extra resolution?
3. Which held-out, matched-budget LoRA intervention would actually test whether
   that extra resolution improves long-horizon repair?
