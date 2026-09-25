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
| Moto endpoint replay | `aggregate_summary.json` | 5/5 endpoint gates pass | `PRoot_ENDPOINT_REPLAY_PASS=true` |
| Moto intermediate states | same | 52 states; 44 valid `(F,R)` states | Every state attempted and preserved |
| Moto transitions | same | 34 neutral, 3 positive, 1 negative, 9 invalid-adjacent | Non-monotone path structure exists |
| Tool outcomes | per-task summaries | 48 attempted; 47 successful mutations | One failed editor attempt correctly creates no state |
| Exact-25 batch | `runs/exact25_intermediate_v01/aggregate_summary.json` | 25/25 summaries; 18/25 strict endpoint gates | Partial qualification, not 25 valid endpoints |
| Exact-25 observations | same | 184 states; 119 valid; 65 invalid | Invalid collection is not an `(F,R)` failure count |
| Exact-25 transitions | same | 13 positive, 85 neutral, 2 negative, 59 invalid-adjacent | Two observed regressions; no certificate test |
| Real SWE-smith RL candidate slice | `runs/swesmith_agentic_64_candidates_v01/manifest.json` | 64 official rows, eight images, 48/16 split | Selection only; not 64 validated environments |
| Real Modal gold smoke | `SWE_SMITH_AGENTIC_64_STATUS.md` | One h11 task: clean pass, injected fail, restored pass | One-task execution qualification; no agent rollout or RL |
| Q-first first screen | `runs/swesmith_online_status_v01/summary.json` | 4/311 q-valid | Qualification yield, not RL performance |
| Q-first expanded screen | same, timestamped snapshot | 28/171 q-valid from 724 candidates | Partial running snapshot; no frozen formal set |
| First four q-valid gold checks | same | 4/4 self-consistent, one repository | Engineering qualification only |
| Pinned model GPU smoke | same | 16-token generation; 2.887 GiB peak allocated | Model loading works, not training |
| Restricted-agent one-update smoke | same | terminal q scorer failed before optimizer update | No GRPO update or scientific result |
| Two-arm survival attempt | `runs/swesmith_survival_status_v01/summary.json` | 20/8 frozen; first semantic update: 8/8 invalid edits, tied rewards, gradient 0 | Stopped; no valid test-vs-semantic result or heldout evaluation |
| One-edit body census, fixed partial snapshot | [`BODY_CENSUS_PARTIAL_6_TASKS_20260923.md`](BODY_CENSUS_PARTIAL_6_TASKS_20260923.md) | 6/28 tasks complete; 81/96 structural-valid, 79/96 executable; `q` varies on 6/6 versus public tests on 2/6; `q` splits 197/408 same-test pairs | Descriptive information evidence only; no eight-step or RL result |
| One-edit reward census, later fixed partial snapshot | [`BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md`](BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md) | 26/28 tasks complete; 328/416 executable, 43 solved; semantic reward varies on 25/26 tasks versus test reward on 11/26; 14/26 have test tied but semantic varying | Extra reward resolution only; no trained-policy result |
| Offline fixed-data LoRA, Test vs Semantic | [`STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md`](STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md) | 20 updates/arm; nonzero gradients; held-out fixed-candidate ranking unchanged between arms | One-step side experiment; cannot answer long-horizon RL question or solve-rate change |
| Interrupted open-loop eight-step source capture | same | 438 state rows; 54 trajectories reached P8 across 4 tasks | No public feedback or q at P2–P8; not the agreed coding-agent trajectory |
| Frozen-policy Oracle Credit pilot | [`ORACLE_CREDIT_RESULTS_20260925.md`](ORACLE_CREDIT_RESULTS_20260925.md) | 6 tasks; 96 P1 candidates; 384/384 P8 outcomes; 14 successes; zero confidently distinguishable Q pairs | Complete data, but K=4 future-value comparison is inconclusive; no RL result |

## First replay conclusion

`conan-io__conan.86f29e13.pr_17531.mz3vj24e` is an exact-bound resolved
trajectory. Its pinned upstream top-level patch modifies Moto files, while its
messages and editor actions modify Conan. Replaying the two explicit mutation
calls—not the top-level patch—restores the endpoint test. This qualifies the
structured replay path and independently confirms one declared outcome.

## Moto intermediate audit v0.1

| Trajectory | Declared | Curve summary | Endpoint |
|---|---:|---|---:|
| `pr_5043` | resolved | `(1,0)` repeated, then `(0,0)` | agrees |
| `pr_6509` | resolved | `(1,0) -> (0,0) -> invalid x8 -> (0,0)` | agrees |
| `pr_6557` | unresolved | `(2,0)` throughout 12 successful mutations | agrees |
| `pr_7144` | unresolved | `(4,0) -> (4,12) -> (4,1)` | agrees |
| `pr_7946` | unresolved | `(2,0)` throughout 6 successful mutations | agrees |

Here `F` is the number of nonpassing FAIL_TO_PASS cases and `R` is the number
of nonpassing PASS_TO_PASS cases. `pr_7144` is the first direct negative
transition: an edit to `moto/acm/models.py` introduces 12 regressions; the next
edit reduces them to one but never resolves the target failures. In `pr_6509`,
an edit to `moto/iotdata/responses.py` first reaches `(0,0)`. The following edit
to `moto/iotdata/__init__.py` imports a symbol unavailable from `moto.core`,
causing collection errors for eight states; a later edit restores collection and
the trajectory ends at `(0,0)`. This is a trajectory-induced invalid region, not
a missing replay action or test-induced workspace mutation.

The compact canonical artifact is
[`runs/moto_intermediate_v01/aggregate_summary.json`](runs/moto_intermediate_v01/aggregate_summary.json).
Per-state XML and console logs are retained locally but omitted from the public
review path; the five per-task `summary.json` files preserve every curve.

## What these results do not show

The five-task, one-repository slice does not estimate population prevalence and
does not yet establish that a certificate representation predicts these states.
It does establish that terminal outcomes discard real, decision-relevant path
structure worth auditing. There is still no value-predictor comparison, sample-
efficiency result, additive-decomposition validation, or online-RL result.

## Exact-25 follow-up

The full exact-binding batch finished, but seven task endpoints are invalid
because of test-runner or image-environment problems. A naive 23/25 declared
endpoint agreement counts invalid unresolved endpoints as agreement; the
strict initialized-and-valid endpoint gate passes only 18/25. The aggregate
preserves all 25 task curves and failures. The diagnoses and frozen next steps
are in [`EXACT25_RESULTS_AND_NEXT_v01.md`](EXACT25_RESULTS_AND_NEXT_v01.md).
This remains historical trajectory replay, not live agent evaluation or RL.
