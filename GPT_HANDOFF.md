# GPT handoff: original theory and September 21 plan summarized

- Review base: `30e3039` (completed Oracle Credit evidence and its handoff)
- Evidence head: `a499d71fb9d16d667e62ca7c833e34639130f512`
- This handoff is metadata-only. Review the evidence-head delta first.

## Read first

1. [`THEORY_AND_HANDOFF_SUMMARY_20260925.md`](THEORY_AND_HANDOFF_SUMMARY_20260925.md):
   the concise synthesis of the September 24 theory recheck, original September
   21 experiment handoff, and certificate-compatible RL adapter.
2. [`ORACLE_CREDIT_RESULTS_20260925.md`](ORACLE_CREDIT_RESULTS_20260925.md):
   unchanged completed SWE-smith pilot and its inconclusive result.
3. [`docs/CODEX_CERTIFICATE_AGENTIC_RL_EXPERIMENT_PLAN_v01.md`](docs/CODEX_CERTIFICATE_AGENTIC_RL_EXPERIMENT_PLAN_v01.md)
   and [`baselines/prior/certificate_rl_adapter/THEORY.md`](baselines/prior/certificate_rl_adapter/THEORY.md):
   original long-form proposal and committed adapter source, only if detail is
   needed. Do not open large per-state JSONL first.

## Decision-relevant delta

This update **adds no new experiment or code change**. It makes the theory
boundary explicit: paired semantic certificates or frozen 256-probe agreement
measure *present program behavior*, not certified future agent value.
Objective-preserving potential shaping with correctly matched `Phi+W` yields
the same TD/GAE signal as the original reward plus total critic; a denser
reward display alone cannot rescue tied whole-trajectory GRPO groups without
changing the objective. The original handoff's proposed critic baselines,
leakage checks, task-family splits and online gate are research design, not
completed outcomes; its default online budget was zero.

The adapter's finite-tree algebra and the September 24 independent finite-MDP
recheck are reported as numerical identity checks, **not** SWE execution or
trained-RL performance. The current SWE-smith `q` is a behavioral probe proxy,
not the proposal's learned interval-certified potential. The completed Oracle
Credit pilot, its K=4 uncertainty and the absence of a trained long-horizon
RL comparison are unchanged from the review base.

## Reviewer questions

1. Does the synthesis keep static semantics, edit-level certificates and
   policy-conditioned continuation value adequately separate?
2. Are any claims in the original September 21 plan inadvertently presented
   as implemented or empirically established?
3. Does the current Oracle Credit result support any stronger conclusion than
   the explicitly scoped inconclusive frozen-policy P8 pilot?
