# SWE-smith replay qualification v0.1

## Verdict

**Engineering qualification passes for one exact-bound resolved trajectory under
udocker/PRoot. This is not yet the formal 40-trajectory scientific audit.**

The initialized task reproduces its target failure (`1 failed, 1 passed`).
Fail-closed replay of the two explicit `str_replace_editor` mutations restores
the target tests (`2 passed`). This agrees with the trajectory's declared
`resolved=true` label.

## Frozen identity

- Trajectory dataset: `SWE-bench/SWE-smith-trajectories`
- Trajectory revision: `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`
- Trajectory: `conan-io__conan.86f29e13.pr_17531.mz3vj24e`
- Task dataset: `SWE-bench/SWE-smith-py`
- Task revision: `77cab9055d42ab4a5c25c89a8f937096db13558e`
- Binding: exact task match
- Image: `swebench/swesmith.x86_64.conan-io_1776_conan.86f29e13:latest`
- Docker Hub manifest digest:
  `sha256:db8589f5c3994d4c1559118778f8730af62626bf053d53f1a35c5ea2710074a2`
- Image repository HEAD: `74a5787ce5a69519232adebde844c3f480ce9aad`
- Runtime: udocker 1.3.17, udockertools 1.2.11, P2 execution mode

## Evidence sequence

1. The clean image passes both selected tests: `2 passed`.
2. Applying the exact official task bug patch produces the intended endpoint:
   `test_replace_in_file` fails while the control test passes.
3. Replay applies exactly two explicit mutations from the selected rollout:
   creation of `/testbed/reproduce.py`, followed by one unique source
   replacement in `conan/tools/files/files.py`.
4. The same endpoint tests then pass: `2 passed`.

Primary artifacts are `task_initialized_tests.txt`, `replay_receipt.json`, and
`after_replay_tests.txt`. The two diff files record initialized and post-replay
source state.

## Upstream integrity finding

The selected upstream row's top-level `patch` field is not the Conan agent
patch. At the exact pinned row and exact `traj_id`, it starts with
`moto/secretsmanager/models.py` and has SHA-256
`e7a5298bd378abcc1193bd78cb9c9b19f7a9bdc52164ab13fcd9d4b3ff94846d`.
The messages and editor calls are Conan-specific. Therefore the audit must
reconstruct edits from structured tool calls; it must not trust the top-level
`patch` field as replay input.

## Limits and next gate

- This run qualifies the no-root server path; PRoot is not equivalent to the
  official Docker harness. Cross-validation on standard Docker remains required
  before treating replay discrepancies as scientific evidence.
- Only one resolved trajectory is qualified. No claim about intermediate-state
  prevalence or resolved/unresolved differences is supported yet.
- The frozen panel contains 25 exact task bindings and 15 repo-profile
  fallbacks. Formal primary analysis should begin with the 25 exact bindings;
  fallbacks remain a separately reported sensitivity stratum.
