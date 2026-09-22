# Results

## Canonical status table

| Item | Evidence | Result | Interpretation |
|---|---|---:|---|
| Frozen panel | `selection_manifest.json` | 40 trajectories; 20/20 outcomes; 24 repos | Selection complete |
| Structured edits | `structural_summary.json` | 200 edits; median 3 per trajectory | Selection-conditioned only |
| Task binding | `task_bindings_manifest.json` | 25 exact; 15 profile fallback | Primary replay must use exact stratum |
| Patch alignment | `patch_alignment_summary.json` | 4 aligned, 16 disjoint, 11 empty, 9 without non-create mutations | Top-level patch is not trusted |
| Resolved path alignment | same | 1/13 = 7.69% among resolved rows with non-create edits | Necessary integrity check fails broadly |
| Conan clean image | `clean_image_tests.txt` | 2 passed | Environment sanity |
| Conan initialized task | `task_initialized_tests.txt` | 1 failed, 1 passed | Target failure reproduced |
| Conan structured replay | `after_replay_tests.txt` | 2 passed | Matches `resolved=true` |

## First replay conclusion

`conan-io__conan.86f29e13.pr_17531.mz3vj24e` is an exact-bound resolved
trajectory. Its pinned upstream top-level patch modifies Moto files, while its
messages and editor actions modify Conan. Replaying the two explicit mutation
calls—not the top-level patch—restores the endpoint test. This qualifies the
structured replay path and independently confirms one declared outcome.

## What these results do not show

The current counts do not establish that a useful certificate intermediate
exists. They also do not compare value predictors, demonstrate sample
efficiency, validate additive value decomposition, or authorize online RL. The
next scientific unit is a balanced exact-bound replay subset with pre/post-edit
observations and a frozen intermediate-state definition.
