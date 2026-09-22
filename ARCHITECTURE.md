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
4. Resolve all paths inside `/testbed`; reject traversal or external paths.
5. `str_replace` must match exactly once. Ambiguity is an error, not a heuristic.
6. Retain create operations in the receipt even when their files are untracked.
7. Run target and control tests before and after replay; preserve both outputs.
8. Never substitute the upstream top-level `patch` for tool-call replay when the
   integrity audit is empty or disjoint.
9. Treat PRoot as engineering qualification. Cross-validate decision-changing
   results on official Docker.

## Source map

| File | Responsibility |
|---|---|
| `scripts/select_swesmith_tool_trajectories.py` | deterministic balanced selection and edit-boundary extraction |
| `scripts/fetch_swesmith_task_metadata.py` | pinned task binding with explicit fallback labels |
| `scripts/audit_swesmith_patch_alignment.py` | patch/editor path integrity classification |
| `scripts/replay_structured_edits.py` | confined, fail-closed mutation replay and receipts |
| `scripts/summarize_swesmith_selection.py` | descriptive, selection-conditioned panel summary |
| `tests/` | unit checks for selection, integrity audit, replay, and prior manifests |

## Planned next slice

The next bounded slice is the single Moto image containing five exact selected
tasks (two resolved, three unresolved). It should record state immediately before
and after every source mutation. The primary analysis must be frozen before
inspecting outcome-conditioned intermediate patterns.
