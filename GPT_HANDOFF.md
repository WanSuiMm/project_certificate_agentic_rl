# GPT handoff: body-action census and queued eight-step observation

- Review base: `e812e33dca965f8e0d9793f4e431575f6372b314`
- Evidence head: `ed791962a784d07efa989aca301b1bef43002946`
- This handoff is metadata-only. Review the evidence-head delta; historical
  exact-25 replay claims and the stopped GRPO outcome are unchanged.

## Read first

1. [`BODY_CENSUS_PARTIAL_6_TASKS_20260923.md`](BODY_CENSUS_PARTIAL_6_TASKS_20260923.md):
   the only new numerical evidence, explicitly a 6/28 provisional snapshot.
2. [`RESULTS.md`](RESULTS.md) and [`GPT_CONTEXT.md`](GPT_CONTEXT.md): formal
   status and claim boundary.
3. For implementation review: `scripts/continue_swesmith_body_trajectories.py`,
   `scripts/grade_swesmith_trajectory_q.py`, `scripts/run_swesmith_trajectory_pipeline.py`,
   `scripts/swesmith_agent_edit.py`, and `scripts/swesmith_agent_worker.py`.

Raw task rows, completions, source states, logs, checkpoints, and private
machine-specific launch receipts are intentionally not in this repository.

## Decision-relevant delta

The apparent 1/4 valid-action body smoke was an indentation-assembler bug:
replaying the same four frozen completions with the fixed assembler yields
4/4 structurally valid programs. All four scored in the isolated evaluator.
The 1.5B base-policy census uses 28 frozen tasks × 16 independent one-edit
samples. Its first six complete tasks yielded 79/96 executable candidates;
public-test pass fraction varies on 2/6 tasks, while reference agreement q
varies on 6/6. Within-task q differs for 197/408 pairs with tied public-test
fraction. These dependent pairs demonstrate additional discrimination, not
long-horizon credit or policy improvement. Some unresolved candidates have
q=1 on the finite observation bank.

A dependent observational run has been dispatched but has no result yet. It
reconstructs each frozen P1 exactly, continues it with seven further model
edits using only public-test feedback, and saves full target-file source/body
and score after each step. Only after all 448 trajectories reach P8 does a
separate worker measure q at shared P0 and each P4/P8. No q is passed to the
online policy; no RL update is part of this run. Local unit tests: 50 passed;
one frozen-task reconstruction was 16/16 hash-exact; one public-only P2 and
one q-only isolated scoring call completed.

## Reviewer questions

1. Does the continuation genuinely keep q out of generation and online
   feedback, and fail closed on any P1 reconstruction mismatch?
2. Are the 6-task tie statistics presented strictly as within-task descriptive
   evidence rather than independent-sample inference or an RL result?
3. When P0/P4/P8 observations complete, what task-level statistic best tests
   whether q changes while public feedback and terminal success remain flat?
