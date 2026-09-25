# Original theory and handoff: what carries into the current project

This is a reading map for three local source packages, **not** a new theorem,
new SWE run, or RL-training result. The September 21 handoff and adapter are
already represented in this repository; the September 24 recheck is summarized
here rather than uploading a second archive of overlapping materials.

| Source | Repository route | What it establishes |
|---|---|---|
| `CODEX_CERTIFICATE_AGENTIC_RL_HANDOFF_20260921` | [`docs/CODEX_CERTIFICATE_AGENTIC_RL_EXPERIMENT_PLAN_v01.md`](docs/CODEX_CERTIFICATE_AGENTIC_RL_EXPERIMENT_PLAN_v01.md), [`docs/HANDOFF_PROVENANCE.md`](docs/HANDOFF_PROVENANCE.md) | A proposed staged experimental contract, not a completed agent/RL experiment. |
| `certificate_rl_adapter` | [`baselines/prior/certificate_rl_adapter/THEORY.md`](baselines/prior/certificate_rl_adapter/THEORY.md), [`check_adapter.py`](baselines/prior/certificate_rl_adapter/check_adapter.py), [`results.json`](baselines/prior/certificate_rl_adapter/results.json) | Potential/critic algebra, counterexamples, and finite-tree numerical checks. |
| `CERTIFICATE_RL_THEORY_RECHECK_20260924` | This summary; original local ZIP SHA-256 `fc692928443da4ddf8a6d125827c1a1e1933585cb586743a3d91112eb7e7105b` | Independent finite-MDP identity recheck and sharper claim limits. Its original script/results remain in the local source package; their numerical values below are reported from that package, not rerun here. |

## The objects must not be collapsed

The original proposal starts with program-task error `E(P)` under a *fixed*
specification and input measure. A paired certificate bounds the **current
edit's** effect, `E(P)-E(P')`. A single-valued semantic potential can be set
to `q(P)=-E(P)` (or the affinely shifted agreement `1-E(P)`). Neither an edit
certificate nor current program agreement is, by itself, the frozen policy's
future success value `V^pi(h)` or action advantage `A^pi(h,a)`. The agent state
`h` also includes history, observations, and remaining budget. A semantic
regression can precede a valuable refactor, and a locally better program can
be harder for a particular policy to finish.

The later SWE-smith implementation uses a different, narrower measurement:
`q(P)` is agreement with a clean reference on a **frozen 256-input probe bank**.
It is an executable behavioral proxy, not the handoff's learned,
interval-certified `q_phi`, and not a proof of correctness on all inputs.
Keeping this distinction is essential when interpreting the current results.

## What the RL algebra does and does not buy

For a frozen potential `Phi(h)` with **every true terminal potential set to
zero**, potential shaping uses

```text
r'_t = r_t + gamma Phi(h_{t+1}) - Phi(h_t)
G'_t = G_t - Phi(h_t)
A'^pi(h_t,a_t) = A^pi(h_t,a_t).
```

For any matched approximate critics `Vhat = Phi + What`, the two TD residuals
and hence GAE values are exactly identical on the same trajectory:

```text
r'_t + gamma What(h_{t+1}) - What(h_t)
  = r_t + gamma Vhat(h_{t+1}) - Vhat(h_t).
```

This is a **reparameterization**, not a new source of policy-gradient
information. If a certificate helps, it must improve finite-data value
estimation, representation, optimization, or exploration under an explicitly
tested implementation. Merely making intermediate reward numbers nonzero does
not prove credit assignment improved. With terminal compensation, trajectories
sharing the same start and terminal outcome do not acquire distinct total
returns for whole-trajectory GRPO solely from potential shaping.

The adapter and recheck also supply counterexamples: an edit with negative
`Delta q` can have positive long-term advantage; arbitrary future-dependent
reward redistribution can preserve total return yet break ordinary
reward-to-go gradients; action-dependent baselines need a correction term; and
`R=Y+0.5q_T` can prefer a policy with *lower* solve probability even though
each successful sample scores above each failed sample. That last reward is a
changed objective, not an objective-preserving potential transformation.

These identities assume correct done/truncation handling, no future leakage,
and a frozen `Phi` along the actor update. They do not make a clipped PPO
implementation, learned critic, or incomplete program verifier automatically
correct or sample-efficient.

## What was checked numerically

- The original adapter reports 500 random finite decision trees, 7,164
  enumerated trajectories, return/TD/GAE and expected-gradient identities at
  floating-point precision, and independent finite-difference gradient error
  below `5.5e-12`. See its committed [`results.json`](baselines/prior/certificate_rl_adapter/results.json).
- The September 24 independent recheck reports 300 finite trees and 4,880
  enumerated trajectories; its return, matched-critic TD/GAE, and expected
  score-function-gradient discrepancies were at approximately `1e-16`, with
  finite-difference error below `1e-11`.
- Neither audit trained an RL agent or executed SWE tasks. Floating-point
  enumeration checks algebra; it does not establish a performance gain.

## The original experimental contract

The September 21 handoff proposed a staged route: reproduce prior finite
artifacts; build a real tool-using agent and a deliberately bounded, audited
program semantics; collect frozen-policy multi-turn trajectories; compare
critics on identical held-out data; **only after gates and an explicit positive
online budget**, consider PPO. Its default online budget was zero.

The critic comparison separated three questions:

1. Does certificate information help beyond return-only and public-test
   feedback (`B0`/`B2` versus certificate methods)?
2. Is the fixed `Phi+W` decomposition better than feeding the *same* certificate
   representation to a flexible critic (`CERT_DECOMP` versus `CERT_FEATURE`)?
3. Does any offline value gain translate to online solve-rate or efficiency gain
   at matched interaction **and** total compute cost?

The plan also required a grouped split by task family/template,
actor/critic visibility and leakage audits, real tool observations, independent
held-out return prediction, and repeated continuations as a *noisy* reference
for value—not as certified ground truth. These were proposed controls and
thresholds, not results already achieved or a mandate to spend on PPO now.

## Current status against that contract

The current [one-edit census](BODY_CENSUS_REWARD_SNAPSHOT_26_TASKS_20260924.md)
shows extra reference-proxy resolution. The completed
[Oracle Credit pilot](ORACLE_CREDIT_RESULTS_20260925.md) tests a smaller and
different question: whether static P1 probe agreement orders P8 success under
a frozen seven-more-edit policy. It completed 384 continuations but produced no
confidently distinguishable candidate-value pair under its K=4 rule; its
descriptive informative-task results did not favor raw q overall. It did **not**
train the original proposal's certificate-supervised critic, validate a
general paired verifier, or test online PPO/GRPO improvement. Thus neither a
positive nor a negative result on the full handoff hypothesis follows.

The durable lesson of all three source packages is narrower and useful:
**program semantics can measure present behavior; future agent value additionally
depends on policy-conditioned continuation dynamics.** Treat q as a possible
input/prior for value estimation, not as certified long-horizon credit.
