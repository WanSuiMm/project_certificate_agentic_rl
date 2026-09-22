# Gate 0: Does the Certificate Intermediate Exist?

**Status:** frozen before implementation
**Date:** 2026-09-21
**Online-RL budget:** zero

## 1. Decision

Before building a full Agentic RL stack, determine whether certificate supervision
contains value-predictive information on real tool-agent trajectories beyond sparse
return labels and public tests.

The primary comparison is certificate-informed held-out return prediction. Runtime
engineering, certificate coverage, and q-fit diagnostics are prerequisites or
mechanism checks, not substitute success criteria.

## 2. Stage A: one vertical slice

Build one isolated 3-6 file Python task with closed finite semantics. A frozen local
LLM must inspect files, invoke actual tools, modify code, and terminate. Save:

- exact task/spec/actor/config hashes;
- every pre-action history and workspace snapshot hash;
- the genuine tool call and observation;
- the old/new program pair for each edit;
- paired certificate interval and analyzer status;
- public-test observation and independently evaluated terminal reward.

The chain passes only if it replays deterministically at the tool layer, the private
terminal evaluator is invisible to the actor, and an independent finite interpreter
contains the exact semantic delta inside every emitted interval.

Stop immediately on certificate containment failure, hidden-test leakage, actor
access to certificate labels, non-replayable snapshots, or fake/scripted actions
substituted for LLM tool use.

## 3. Stage B: minimum signal bank

After Stage A passes, freeze the actor and collect a bounded preflight:

- at least 8 independent training tasks from at least 3 semantic templates;
- held-out development/test tasks grouped by semantic template source;
- at least 32 informative real-agent program-version pairs;
- at least 10 certified improvements and 10 certified regressions;
- all edits retained, including unsupported, unchanged, invalid, and failed edits.

Task generation and model calibration may use smoke/development tasks once. They
must freeze before held-out outcomes are inspected.

## 4. Minimal learning comparison

Use one frozen encoder and capacity-matched heads. Train three seeds for:

- `B0 RETURN_ONLY`: terminal-return labels only;
- `B2 TEST_FEATURE`: public-test supervision plus return prediction;
- `B3 CERT_FEATURE`: certificate-pretrained q exposed as a free value feature;
- `M CERT_DECOMP`: the same frozen q used in `V = beta*q + W`.

B3 and M must reuse the identical q checkpoint. All methods receive the same actor
trajectories, program/spec/history inputs, split, and return labels. Report extra
certificate CPU time and q-training cost separately.

## 5. Primary metric and interpretation

The primary metric is paired task-group macro Brier/MSE on frozen held-out
pre-action prefixes. Prefixes from one trajectory are not independent samples.

- B3 or M beating B0 and B2 supports existence of a useful certificate intermediate.
- M beating B3 additionally supports the fixed additive consumption rule.
- B3 positive but M not better than B3 supports information value only.
- Coverage without held-out return-prediction gain is not a pass.

This preflight is allowed to return `PROMISING_SIGNAL`, `NO_SIGNAL`, `INSUFFICIENT_SIGNAL`,
or `INVALID`. It does not authorize a paper-strength claim.

## 6. Stop and expansion rules

- `INVALID`: repair the runtime and regenerate affected data; do not interpret it.
- `INSUFFICIENT_SIGNAL`: stop model scaling; report missing denominators or policy-task
  mismatch.
- `NO_SIGNAL`: stop the project before LoRA critic or PPO work.
- `PROMISING_SIGNAL`: preregister a larger independent P3 replication. PPO remains
  disabled until the broader G2 criteria and a positive online budget are supplied.

Stage A should fit within several engineering hours. Stage B should be bounded to
roughly 1-2 working days and must not silently expand into the full P0-P4 plan.
