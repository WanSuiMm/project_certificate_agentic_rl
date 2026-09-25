# project_certificate_agentic_rl

- Project: Certificate-Supervised Agentic Value Learning (CSAVL; working name)
- Activity tier: 1
- Lifecycle status: Oracle Credit pilot, offline P2 q, and local grouped-residual audit complete; future-value comparison unresolved at K=4; training remains deferred
- Current artifact: completed 6 × 16 × 4 frozen-policy Oracle Credit pilot with first-hit recount and P2 semantic-drift diagnostic, alongside the earlier frozen panel and exact-bound replays
- Closest venue: ICLR/NeurIPS candidate only if the real-agent intermediate signal survives
- Last verified: 2026-09-25

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

The later open-loop, source-only eight-step continuation was stopped at 54
complete P1–P8 trajectories (of 448 planned), without P2–P8 tests or `q`
values. It is not the agreed closed-loop agent protocol. Fixed-data LoRA
optimization completed but only measures one-step candidate preferences; it
does not test long-horizon semantic credit. The corrected public-feedback
pilot and two-arm whole-trajectory GRPO are implemented but unrun. The five exact Moto
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

- Read first: `ORACLE_P2_POLICY_DYNAMICS_20260925.md`, then `ORACLE_CREDIT_RESULTS_20260925.md` and `ORACLE_CREDIT_BENCHMARK_v01.md`
- Local exploratory DWR result: `DWR_RESIDUAL_AUDIT_20260925.md`; collector `scripts/audit_dwr_residual_local.py`, analyzer `scripts/analyze_dwr_residual_local.py`
- Active entry: `scripts/run_swesmith_oracle_credit.py`
- Active config: `configs/swesmith_oracle_credit_6x16x4_v01.json`
- Oracle analysis: `scripts/summarize_swesmith_oracle_credit.py`
- Failed-run recovery: `scripts/launch_swesmith_oracle_credit.py --import-run-dir FAILED_RUN_DIRECTORY` creates a new, provenance-linked run without changing the failed run
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

The `ORACLE_CREDIT_BENCHMARK_v01.md` pilot completed: six task IDs, all 16
frozen P1 candidates, four closed-loop continuations each through P8. All 384
endpoints are present, but none of the noisy K=4 candidate-value comparisons
passed its confident-pair threshold. `ORACLE_CREDIT_RESULTS_20260925.md` is
the canonical result; this is not evidence that q improves long-horizon RL.
No further paid run is authorized by this status document alone.

### Historical execution notes (superseded by the oracle pilot)

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
in `p_T` and `q`, and whether `q` breaks public-test ties. The first 26
complete tasks are reported provisionally in
`BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md`; the 28-task aggregate remains
pending. A streaming continuation has been dispatched: each complete 16-sample
P1 task block is immediately continued to eight sequential edits, without
waiting for the entire census. The policy sees only public-test feedback.
Every P0–P8 program state is saved; after all trajectories finish, a separate
worker measures q at every state (3612 observations before source-hash
deduplication). This is observational sampling, not RL training. Test-arm training
and heldout evaluation have not run.
See `runs/swesmith_survival_status_v01/summary.json` for the earlier failure.
This agent is restricted to replacing one
function/method with public-test feedback; it is not a full SWE-agent with
arbitrary repository tools. Do not promote the smoke or original 64 candidate
slice into a formal RL result. See `SWE_SMITH_AGENTIC_64_STATUS.md`.
