# project_certificate_agentic_rl

- Project: Certificate-Supervised Agentic Value Learning (CSAVL; working name)
- Activity tier: 1
- Lifecycle status: Moto intermediate audit complete; learning Gate 0 not run
- Current artifact: frozen 40-rollout panel plus five-task exact Moto edit-level replay
- Closest venue: ICLR/NeurIPS candidate only if the real-agent intermediate signal survives
- Last verified: 2026-09-22

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

The project currently has no real-agent learning result. The five exact Moto
rollouts now pass endpoint replay self-consistency under the PRoot execution
substrate. Across 52 states, one unresolved rollout has a genuine regression
excursion `(4,0) -> (4,12) -> (4,1)`, while one resolved rollout reaches `(0,0)`,
passes through a trajectory-induced uncollectable region, and recovers to
`(0,0)`. This is first evidence that terminal labels hide useful path structure,
but it is a single-repository slice and not a prevalence estimate.
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
- Structured replay: `python scripts/replay_structured_edits.py --help`
- Full experiment command: NOT_IMPLEMENTED
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

Do not use the trajectory dataset's top-level `patch` as replay input. Extend the
same frozen edit-level audit to a small repository-diverse exact-bound slice and
separate valid `(F,R)` transitions from invalid collection regions. In parallel,
define the cheapest certificate-derived feature that can be computed on these
frozen states and tested without training a critic. Do not implement PPO before
this gate.
