# project_certificate_agentic_rl

- Project: Certificate-Supervised Agentic Value Learning (CSAVL; working name)
- Activity tier: 1
- Lifecycle status: exact-25 offline replay complete; q-first 20/8 slice frozen; real-task RL stopped after one zero-gradient semantic update; no valid test-versus-semantic result
- Current artifact: frozen 40-rollout panel plus 25 exact-bound edit-level replays
- Closest venue: ICLR/NeurIPS candidate only if the real-agent intermediate signal survives
- Last verified: 2026-09-23

## Research Contract

- Core phenomenon: program-semantic certificate supervision may provide a useful
  intermediate representation for predicting the terminal success of a frozen
  coding-agent policy.
- Sole primary Gate 0 question: on real tool-agent trajectories, does certificate
  supervision improve held-out return prediction beyond return-only and public-test
  baselines?
- Main metric: paired task-group macro Brier/MSE difference on frozen held-out
  pre-action prefixes.
- Hidden mechanism: sound paired semantic intervals constrain a task-conditioned
  potential `q(P, S, mu)` that captures progress information not recoverable as
  efficiently from sparse terminal returns or public tests alone.
- Competing explanations: extra model capacity, extra compute, public-test leakage,
  task/template memorization, easier certificate-supported edits, immediate-error
  correlation without future-value information, and synthetic-task shortcuts.
- Failure / kill criteria: stop before online RL if the runtime chain is invalid;
  fewer than 32 informative real-agent version pairs or fewer than 10 certified
  changes in either direction are observed; `q` collapses to a near-constant; or
  certificate methods fail to improve over both return-only and public-test
  controls on frozen held-out task groups.

## Claim Boundary

The project currently has no live-agent continuation or learning result. The five exact Moto
rollouts now pass endpoint replay self-consistency under the PRoot execution
substrate. Across 52 states, one unresolved rollout has a genuine regression
excursion `(4,0) -> (4,12) -> (4,1)`, while one resolved rollout reaches `(0,0)`,
passes through a trajectory-induced uncollectable region, and recovers to
`(0,0)`. This is first evidence that terminal labels hide useful path structure,
but it is a single-repository slice and not a prevalence estimate.
The subsequent exact-25 batch records 184 states, 119 valid `(F,R)` states,
and two valid negative transitions across different repositories. Only 18/25
tasks meet the strict endpoint gate; seven have invalid test observations from
runner or environment faults. This is a partial engineering qualification,
not a certificate-prediction result or a population estimate.
The copied finite-world
artifacts establish algebraic correctness, counterexamples, and certificate
coverage in a 64-state generated setting. They do not establish critic gain,
online RL gain, general Python verification, or system speedup.

Gate 0 is a controlled coding-agent experiment over a deliberately closed Python
semantics. Even a positive result would support a scoped intermediate-signal claim,
not a claim about arbitrary repositories or general software engineering agents.

## Canonical Entry Points

- Read first: `GATE0_INTERMEDIATE_SIGNAL_PROTOCOL_v01.md`
- Project overview: `README.md`
- Original execution specification: `docs/CODEX_CERTIFICATE_AGENTIC_RL_EXPERIMENT_PLAN_v01.md`
- Prior-evidence provenance: `docs/HANDOFF_PROVENANCE.md`
- Integrity smoke: `python scripts/audit_prior.py`
- Gate 0 config: `configs/gate0_signal.yaml`
- Frozen SWE-smith selection: `runs/selection_v01/selection_manifest.json`
- Task-binding manifest: `runs/selection_v01/task_bindings_manifest.json`
- Patch-integrity audit: `runs/patch_alignment_v01/patch_alignment_summary.json`
- Replay qualification: `runs/qualification_v01/QUALIFICATION_SUMMARY.md`
- Moto protocol: `MOTO_INTERMEDIATE_PROTOCOL_v01.md`
- Moto aggregate: `runs/moto_intermediate_v01/aggregate_summary.json`
- Exact-25 result and next gate: `EXACT25_RESULTS_AND_NEXT_v01.md`
- Exact-25 aggregate: `runs/exact25_intermediate_v01/aggregate_summary.json`
- Exact-25 aggregate generator: `python scripts/summarize_exact25_intermediate.py --run-dir RUN_DIR --expected-tasks 25`
- New exploratory RL protocol: `ORACLE_PROXY_GRPO_SURVIVAL_v01.md`
- Frozen RL config: `configs/oracle_proxy_grpo_survival_v01.json`
- RL runtime and prerequisites: `FUNCTION_SWE_RUNTIME.md`
- Real SWE-smith agentic pivot and 64-task candidate slice: `SWE_SMITH_AGENTIC_64_STATUS.md`
- q-first selector and official-image worker: `scripts/build_swesmith_qfirst_pool.py`, `scripts/swesmith_q_qualifier_worker.py`
- Restricted real-repository one-body-edit trainer: `scripts/train_swesmith_agent_grpo.py`
- Frozen base-policy census (one pass, 28 tasks × 16 candidates): `scripts/sample_swesmith_body_28x16.py`
- Eight-step observational continuation and offline q: `scripts/continue_swesmith_body_trajectories.py`, `scripts/grade_swesmith_trajectory_q.py`; dependency runner `scripts/run_swesmith_trajectory_pipeline.py`
- Modal isolation smoke: `python scripts/smoke_modal_function_swe.py --receipt NEW_RECEIPT_PATH`
- Modal smoke result: `runs/modal_sandbox_smoke_v01/SMOKE_SUMMARY.md`
- RL trainer: `python scripts/train_oracle_proxy_grpo.py --help`
- RL code checks: `python -m unittest discover -s tests -v`
- Structured replay: `python scripts/replay_structured_edits.py --help`
- Restricted experiment entry: `scripts/train_swesmith_agent_grpo.py`; formal GRPO comparison not launched
- Results: engineering qualification only; no learning result

## Evidence State

- Frozen evidence: copied prior adapter and finite semantic-certificate source,
  reports, and compact results under `baselines/prior/`, protected by a manifest.
- Exploratory evidence: the frozen 40-rollout panel contains 25 exact current-task
  bindings and 15 repo-profile fallbacks. Only 1 of 13 declared-resolved rollouts
  with a non-create editor mutation has any edited-path overlap with its top-level
  `patch`; 7 are disjoint and the remainder have empty patches. This is an
  integrity finding, not yet a semantic correctness estimate.
- Qualification evidence: one exact Conan rollout and all five exact Moto
  rollouts reproduce their initialized target failures and agree with declared
  endpoints under fail-closed replay.
- Exploratory scientific evidence: the Moto slice contains one negative valid
  transition and one recoverable invalid region; terminal success alone hides
  both. This motivates, but does not establish, certificate predictive value.
- Exact-25 qualification: batch 25/25 complete, strict endpoint gate 18/25;
  65/184 state observations are invalid. These seven failed tasks remain in the
  denominator pending targeted runner/environment repair.
- Negative results: in the 64-state prior benchmark, exact suffix caching was faster
  and more informative than the abstract paired certificate.
- Unsupported claims: certificate-supervised critic improvement, fixed-decomposition
  advantage, sample-efficiency gain, online PPO gain, and total-compute advantage.

## Reusable Assets

- finite decision-tree potential/GAE algebra audit;
- finite-program paired-certificate generator and independent scalar audit;
- frozen counterexamples against upper-bound subtraction, sign rewards, and
  action-dependent baselines;
- a protocol separating information gain from decomposition gain and online gain.

## Next Action

The user has requested a distinct oracle-proxy GRPO survival experiment, using
only terminal `Y`, `Y+0.5p_T`, and `Y+0.5q(P_T)` rewards. Its protocol is frozen
in `ORACLE_PROXY_GRPO_SURVIVAL_v01.md`. The earlier Function-SWE trainer remains
a surrogate and must not be launched for real SWE-smith tasks. The original
certificate/value-prediction Gate 0 remains unpassed and separate from this
exploratory RL screen.

The latest user direction is to use real SWE-smith repository tasks and RL,
not synthetic Function-SWE samples. The first q-first screen yielded only
4/311 qualifying tasks; the expanded screen stopped with 29/636 q-valid tasks
across three images, and 28 were frozen as 20 train / 8 heldout.
The pinned 1.5B model was downloaded through parallel ranges, verified against
its fixed weight hash, and passed an offline 5090 generation smoke. A
four-task, one-update multi-edit GRPO engineering smoke failed in the terminal
q scorer (`callable_import_or_execution_failed`) before any optimizer update.
Subsequent one-edit attempts reached `optimizer.step()`, but the first formal
semantic update had 8/8 invalid edits, tied within-group rewards, and zero
gradient. A later body-only 1.5B sample initially appeared 1/4 structurally
valid. Replaying the same four frozen completions after fixing an assembler
indentation bug made all 4/4 parse successfully; the former 1/4 figure was a
parser artifact, not evidence that the model lacks valid-action support. All
four then returned terminal `Y`, public-test fraction `p_T`, and reference
agreement `q` from the isolated scorer. On 2026-09-23, a single 1.5B base-policy
28×16 census was dispatched to measure valid action rate, task-level variation
in `p_T` and `q`, and whether `q` breaks public-test ties. The first six
complete tasks are reported provisionally in
`BODY_CENSUS_PARTIAL_6_TASKS_20260923.md`; the 28-task aggregate remains
pending. A dependent run is queued to continue the same 448 first edits to
eight sequential edits with public-test feedback only. Every program state is
saved; only after all trajectories finish will a separate worker measure q at
P0/P4/P8. This is observational sampling, not RL training. Test-arm training
and heldout evaluation have not run.
See `runs/swesmith_survival_status_v01/summary.json` for the earlier failure.
This agent is restricted to replacing one
function/method with public-test feedback; it is not a full SWE-agent with
arbitrary repository tools. Do not promote the smoke or original 64 candidate
slice into a formal RL result. See `SWE_SMITH_AGENTIC_64_STATUS.md`.
