# Oracle-proxy GRPO survival experiment v0.1

Status: protocol and reward contract frozen on 2026-09-23; implementation has
passed a trusted local toy smoke, but **no formal run has launched**.
This is a new, exploratory oracle-reward experiment. It does not retroactively
pass or replace the project's earlier certificate/value-prediction Gate 0.

Question: with a privileged clean reference available at training time, does
its behavioral agreement reward improve a small coding policy's held-out solve
rate more than an equally weighted public-test reward?

Use the exact configuration in
[`configs/oracle_proxy_grpo_survival_v01.json`](configs/oracle_proxy_grpo_survival_v01.json):
Qwen2.5-Coder-1.5B-Instruct, BF16 LoRA rank 16, 8K context, three edit actions,
four rollouts per task, 48 training and 16 held-out tasks, one seed, 100 updates
per arm. Each update samples four tasks. Arms start from the same base checkpoint
and receive equal rollout/update/evaluation budgets.
The implementation config also fixes AdamW learning rate `1e-5`, two clipped
policy epochs per update, clip epsilon `0.2`, frozen-reference KL coefficient
`0.04`, gradient norm cap `1.0`, and unwarped sampling (`temperature=top_p=1`).

For task-specific fixed input bank (X), compute the *unchanged* proxy

\[q(P)=|X|^{-1}\sum_{x\in X}{\bf1}[Obs(P,x)=Obs(P^\star,x)].\]

`Obs` is the return value or exception type. Freeze 256 inputs and reference
outputs per task before training. Reward only the terminal program:

- A: `Y`.
- B: `Y + 0.5 p_T`, where `p_T` is the final public-test pass fraction.
- C: `Y + 0.5 q(P_T)`.

`Y` comes from the task's terminal verifier. No headroom closure, editwise
delta, trajectory average, or substitute semantic feature is allowed. The
implementation is `scripts/oracle_proxy_reward.py`; it rejects missing or
out-of-range scores. A success always outranks a failure in every arm.

Task qualification is a necessary execution prerequisite, not a new signal
audit: select 64 frozen SWE-smith-derived Python tasks with a single executable
target function/module, supported primitive inputs, deterministic reference and
tests, nonzero initial reference disagreement, and a safe isolated candidate
executor. This creates a Function-SWE surrogate; its terminal `Y` is exact
public-plus-hidden function-case success, **not** an official full-repository
SWE-smith solve.
Freeze task IDs, source hashes, input-bank hashes, split, and terminal test
commands *before* seeing RL results. No tasks may be swapped post outcome.

Evaluate each arm at update 0, 20, 40, 60, 80, 100 on the same 16 held-out
tasks, pass@1 by the real test verifier without reference/proxy access to the
agent. Report integer solves out of 16 and the matched rollout count, not just
rounded percentages. Survival criterion: C must beat B by at least two held-out
tasks at update 100 and show an earlier/better solve-rate curve; otherwise do
not escalate. One seed and 16 tasks are a go/no-go screen, not a publication-
strength estimate. Preserve negative results.

Launch gate: a frozen 48/16 task manifest, a working isolated candidate runner,
the exact model checkpoint and training environment, an end-to-end one-update
smoke, and an uncontended GPU. None is inferred from a completed replay batch.
Record host, GPU, PID, command, run directory, code commit, model revision,
task/input hashes, and launch time when actually dispatched. A queued or
prepared configuration must never be reported as training in progress.
Implementation entry points and sandbox boundary: `FUNCTION_SWE_RUNTIME.md`.
