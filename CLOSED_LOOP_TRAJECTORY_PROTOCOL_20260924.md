# Closed-loop SWE-smith trajectory pilot (code ready; not run)

This replaces the open-loop capture at commit `885275c`. The server was off
when this correction was made. No new trajectory or q result is claimed.

## Question and frozen pilot

On eight q-valid SWE-smith tasks (the first eight IDs in the existing frozen
28-task selection), sample 16 independent eight-edit trajectories per task
with the pinned Qwen2.5-Coder-1.5B-Instruct policy. The task is the primary
coverage unit; the 16 trajectories are repeated samples within a task.

At P0, run public tests on the officially injected buggy program. For each
edit t=1,…,8, the policy sees the current source, issue, target function, and
the previous public-test feedback. Apply the body edit, run public tests on
the resulting state, save source/hash/action/feedback/public result, and use
that feedback at the next step. Invalid edits leave source unchanged and are
explicitly identified in the next prompt. No q bank observations or q scores
are placed in a policy prompt or online scorer request.

Execution reuses **one network-blocked Modal sandbox and injected checkout per
task** across its 16 trajectories and public-test observations. Candidate
source is fully rewritten before each test. The worker detects non-target
repository mutations by tests/probes and fails closed rather than letting
later observations inherit them. This removes per-state sandbox startup; it
does **not** remove the 1,032 public-test runs or guarantee a wall time.

After **all** pilot trajectories complete, score every P0,…,P8 state with the
unchanged 256-case reference behavioral agreement q. The offline grader
uses one sandbox per task, sends the task's q bank once, deduplicates identical task/source hashes, and
joins q with the stored p and Y
for each observation. Expected rows: 8 shared P0 + 8×16×8 = **1,032**.
The grader rejects open-loop receipts, incomplete captures, altered state
files, or states without public feedback.

Primary observational checks, to be performed only after a real run:

1. Among steps where public p and terminal Y stay unchanged, how often does q move?
2. How often does q change before p along a trajectory?
3. Within task and tied public p at an intermediate step, is q associated with later p or Y?

These are observational diagnostics, not an RL intervention or a proof of
causal credit. A failed/missing environment observation is recorded as invalid,
not silently scored as a semantic failure. Do not expand from eight to 28 tasks
until the pilot's complete state and scorer receipts have been inspected.

## Entry points

- Online capture: [`capture_swesmith_body_trajectories.py`](scripts/capture_swesmith_body_trajectories.py)
- Offline all-state q: [`grade_swesmith_trajectory_q.py`](scripts/grade_swesmith_trajectory_q.py)
- Frozen config: [`swesmith_closed_loop_8x16x8_v01.json`](configs/swesmith_closed_loop_8x16x8_v01.json)

Run capture into a **new** output directory; never resume the stopped
`body_trajectories_capture_only_1p5b_28x16x8_v01` directory. Use the frozen
selection, task and q-bank files from the existing 28-task setup. Then run
the offline grader only after capture reports `complete` and 1,032 states.

The historical 54 open-loop P1–P8 trajectories remain preserved as a
diagnostic snapshot, not as examples of this closed-loop protocol. The
one-edit offline LoRA comparison is a separate, non-decision-relevant side
experiment for the long-horizon question.
