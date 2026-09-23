# GPT handoff: body-action census and streaming eight-step observation

- Review base: `472010b4770ea485a5b286aac31912d63a43da99`
- Evidence head: `33bbac919963021ff12b98f408f55725935e498a`
- This handoff is metadata-only. Review the evidence-head delta for the
  streaming/full-state protocol; historical exact-25 replay claims, the
  provisional six-task statistics, and stopped GRPO outcome are unchanged.

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

A streaming observational run has been dispatched but has no result yet. As
soon as a task's 16 P1 samples are complete, it reconstructs each P1 exactly,
continues with seven further model edits using only public-test feedback,
and saves full target-file source/body and score after every step. After all
448 trajectories reach P8, a separate worker measures q at every P0–P8 state
(3612 observations before per-task source-hash deduplication). No q is passed to the
online policy; no RL update is part of this run. Local unit tests: 50 passed;
one frozen-task reconstruction was 16/16 hash-exact; one public-only P2 and
one q-only isolated scoring call completed. Streaming dispatch is verified;
the first saved P0–P4 states were hash-valid and contained no q.

## Reviewer questions

1. Does the continuation genuinely keep q out of generation and online
   feedback, and fail closed on any P1 reconstruction mismatch?
2. Are the 6-task tie statistics presented strictly as within-task descriptive
   evidence rather than independent-sample inference or an RL result?
3. When full P0–P8 observations complete, how often does q change before
   public feedback, and does early q predict later outcomes within public-test ties?
