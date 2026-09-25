# Architecture and invariants

## Data flow

```text
pinned SWE-smith tool split
        |
        v
balanced selector -----> selection manifest + 40 raw rows
        |
        v
pinned task binder ----> 25 exact + 15 profile-fallback bindings
        |
        +----> top-level patch alignment audit
        |
        v
exact task image + official task mutation
        |
        v
fail-closed structured editor replay
        |
        v
pre/post endpoint tests + replay receipt
```

## Frozen sources

- Trajectories: `SWE-bench/SWE-smith-trajectories` at
  `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`.
- Python tasks: `SWE-bench/SWE-smith-py` at
  `77cab9055d42ab4a5c25c89a8f937096db13558e`.
- Selection seed: `20260921`; split: `tool`; shuffle buffer: `4096`.

## Replay invariants

1. Identify rows by exact `traj_id`, not merely `instance_id`; one task may have
   multiple rollouts.
2. Initialize an exact task with its pinned official task mutation before
   applying agent edits.
3. Replay only explicit `str_replace_editor` mutation calls.
   A call creates a state only when its linked tool observation confirms that
   the edit succeeded; failed attempts remain recorded but do not mutate files.
4. Resolve all paths inside `/testbed`; reject traversal or external paths.
5. `str_replace` must match exactly once. Ambiguity is an error, not a heuristic.
6. Retain create operations in the receipt even when their files are untracked.
7. Run target and control tests before and after replay; preserve both outputs.
8. Never substitute the upstream top-level `patch` for tool-call replay when the
   integrity audit is empty or disjoint.
9. Treat PRoot as the experimental execution substrate and require endpoint
   self-consistency for every replayed task.

## Source map

| File | Responsibility |
|---|---|
| `scripts/select_swesmith_tool_trajectories.py` | deterministic balanced selection and edit-boundary extraction |
| `scripts/fetch_swesmith_task_metadata.py` | pinned task binding with explicit fallback labels |
| `scripts/audit_swesmith_patch_alignment.py` | patch/editor path integrity classification |
| `scripts/replay_structured_edits.py` | confined, fail-closed mutation replay and receipts |
| `scripts/run_moto_intermediate.py` | task initialization and F/R measurement after every successful mutation |
| `scripts/summarize_moto_intermediate.py` | compact endpoint gates, curves, and transition counts |
| `scripts/audit_replay_surface.py` | undo and shell-side mutation audit for exact-bound rows |
| `scripts/summarize_swesmith_selection.py` | descriptive, selection-conditioned panel summary |
| `scripts/build_swesmith_qfirst_pool.py` | outcome-blind function-level candidate pool from pinned tasks |
| `scripts/swesmith_q_qualifier_worker.py` | generic-input reference/bug agreement bank in official image |
| `scripts/batch_swesmith_q_modal.py` | batched Modal q qualification |
| `scripts/freeze_swesmith_q_selection.py` | freeze qualified train/evaluation selection |
| `scripts/swesmith_agent_edit.py` | restricted replacement-edit action parser and confinement |
| `scripts/swesmith_agent_worker.py` | official-image rollout with public-test feedback |
| `scripts/swesmith_modal_executor.py` | remote sandbox scoring interface |
| `scripts/run_swesmith_oracle_credit.py` | frozen-policy P1-to-P8 closed-loop continuation, public-feedback ready queue, durable action/state/outcome journals |
| `scripts/summarize_swesmith_oracle_credit.py` | candidate Monte Carlo Q, Wilson intervals, task-macro p/q comparisons |
| `scripts/export_swesmith_oracle_public.py` | source-free evidence export with raw artifact hashes and row-count checks |
| `scripts/train_swesmith_agent_grpo.py` | pinned-model restricted-agent GRPO trainer; first smoke failed before update |
| `tests/` | unit checks for selection, integrity audit, replay, and prior manifests |

## Completed bounded slice

The single Moto image slice contains five exact selected tasks (two resolved,
three unresolved). It records state immediately before and after every confirmed
source mutation under the frozen protocol in `MOTO_INTERMEDIATE_PROTOCOL_v01.md`.
The next slice should add repository diversity while preserving the same state
definition and endpoint gate.
