# Certificate-Supervised Agentic Value Learning

This project asks whether program-semantic certificates expose a useful
intermediate state for predicting terminal success on real coding-agent
trajectories. The original frozen SWE-smith trajectory audit is complete.
A separate real-task agentic-RL preparation branch has 64 original candidate
tasks, a stopped q-first screen, and an offline 5090 model smoke.
The early two-arm survival attempt stopped after one zero-gradient semantic
update. A later fixed-data LoRA comparison completed 20 updates per arm but
showed no held-out candidate-ranking separation. An eight-step source capture
was stopped with only 54 complete P1–P8 trajectories; P2–P8 were not scored.
See the [stopped snapshot](STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md)
for exact evidence and limits. The earlier
[26-task provisional reward snapshot](BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md)
shows extra reference-proxy discrimination, not an RL improvement.

## Start here

1. [`STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md`](STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md):
   latest offline LoRA result and interrupted eight-step source capture.
2. [`BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md`](BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md):
   provisional reward-information result and exact non-claim boundary.
3. [`SWE_SMITH_AGENTIC_64_STATUS.md`](SWE_SMITH_AGENTIC_64_STATUS.md): current
   real-task pivot, qualification protocol, engineering failure, and open gates.
4. [`GPT_CONTEXT.md`](GPT_CONTEXT.md): compact task definition, claim boundary,
   code map, and reviewer questions.
5. [`RESULTS.md`](RESULTS.md): canonical aggregates and formal status.
6. [`ARCHITECTURE.md`](ARCHITECTURE.md): data flow, invariants, and exact scripts.
7. [`MOTO_INTERMEDIATE_PROTOCOL_v01.md`](MOTO_INTERMEDIATE_PROTOCOL_v01.md):
   the frozen edit-level replay protocol.
8. [`EXACT25_RESULTS_AND_NEXT_v01.md`](EXACT25_RESULTS_AND_NEXT_v01.md):
   full exact-bound audit, invalid cases, and next decision.
9. [`PROJECT.md`](PROJECT.md): broader research contract and next gate.

Separate exploratory RL code: [`ORACLE_PROXY_GRPO_SURVIVAL_v01.md`](ORACLE_PROXY_GRPO_SURVIVAL_v01.md)
defines the frozen three-arm comparison; [`FUNCTION_SWE_RUNTIME.md`](FUNCTION_SWE_RUNTIME.md)
maps its code, task format, and unfulfilled isolation/task prerequisites. It has
not produced a training result. The Modal Sandbox scorer passed a live two-score
toy smoke; the original 64-task manifest is frozen, but the real-task GRPO
update has not passed.
Do not run that Function-SWE trainer on the new repository-level task slice.
For the new slice, start with
[`runs/swesmith_agentic_64_candidates_v01/manifest.json`](runs/swesmith_agentic_64_candidates_v01/manifest.json),
then [`candidate_index.json`](runs/swesmith_agentic_64_candidates_v01/candidate_index.json).
The 64-row raw task JSONL is retained locally and regenerable from the pinned
dataset, but omitted from the public repository.

The two large raw JSONL files are frozen locally but intentionally omitted from
the public review repository because they contain third-party issue text and
machine-specific paths. Their hashes remain in the manifests, and the pinned
selector/binder scripts regenerate them.

## Current result

- Frozen panel: 40 trajectories, balanced 20 resolved / 20 unresolved, from 24
  repositories; all use `claude-3-5-sonnet-20241022`.
- Current task binding: 25 exact tasks and 15 repo-profile fallbacks.
- Integrity finding: only 1 of 13 resolved trajectories with a non-create editor
  mutation has any path overlap between its actual editor calls and the dataset's
  top-level `patch`. Empty or disjoint patches are not valid replay inputs.
- Runtime qualification: one exact Conan task changes from `1 failed, 1 passed`
  after task initialization to `2 passed` after fail-closed structured-edit
  replay, agreeing with `resolved=true`.
- Moto exact slice: all five endpoint replays agree with their declared outcome
  (`PRoot_ENDPOINT_REPLAY_PASS=true`) across 52 recorded states and 47 successful
  filesystem mutations.
- Intermediate signal: one unresolved trajectory moves `(F,R)` from `(4,0)` to
  `(4,12)` and then `(4,1)`; one resolved trajectory reaches `(0,0)`, becomes
  temporarily uncollectable after a bad import edit, and later returns to
  `(0,0)`.
- Scientific status: the slice establishes that endpoint labels hide meaningful
  edit-level path structure. It does not estimate prevalence or show a
  certificate-trained critic or online-RL gain.
- Full exact-bound batch: 25/25 summaries, 184 states, but only 18/25 strict
  endpoint gates pass. Seven invalid cases need targeted runner/environment
  repair; two valid negative transitions are observed. This historical batch
  has no live-agent continuation, critic training, or RL result.
- Real-task q-first screen: 4/311 in the first completed screen; an expanded
  724-candidate screen had 28 qualifying tasks among 171 checked at its
  timestamped partial snapshot. This is selection, not policy performance.
- Fixed-model smoke: pinned Qwen2.5-Coder-1.5B-Instruct loaded and generated
  on RTX 5090. The restricted-agent one-update GRPO smoke failed in terminal
  q scoring before an optimizer update. See
  [`summary.json`](runs/swesmith_online_status_v01/summary.json).
- Survival attempt: 28 tasks frozen as 20 train / 8 heldout across three images.
  The first semantic update had 8/8 invalid edits, tied rewards and zero
  gradient; the run was stopped. No test-arm update or heldout evaluation was
  completed. See [`survival summary`](runs/swesmith_survival_status_v01/summary.json).

## Reproduce compact checks

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/audit_prior.py
python scripts/select_swesmith_tool_trajectories.py --output /tmp/selection_v01
python scripts/audit_swesmith_patch_alignment.py \
  --trajectories /tmp/selection_v01/selected_trajectories.jsonl \
  --output-dir /tmp/patch_alignment_check
python scripts/freeze_swesmith_agentic_candidates.py \
  --output-dir /tmp/agentic64 --cache-dir /tmp/swesmith_hf_cache
python scripts/summarize_swesmith_agentic_candidates.py \
  --tasks /tmp/agentic64/tasks.jsonl --manifest /tmp/agentic64/manifest.json \
  --output /tmp/agentic64/candidate_index.json
```

The replay requires the pinned SWE-smith image. PRoot is the current experimental
execution substrate; each task is initialized from a fresh image-derived rootfs.
