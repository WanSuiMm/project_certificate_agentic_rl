# GPT handoff: q-first SWE-smith RL engineering status

- Review base: `85d88eb692ba753eb539d78d477055e0d7d3a309`
- Evidence head: `ae9b436ec9e5621a7f66a426fcfc12b97e672dde`
- This handoff is metadata-only. Review the evidence-head delta; the historical
  exact-25 replay data and its claim boundary are unchanged.

## Read only these first

1. [`SWE_SMITH_AGENTIC_64_STATUS.md`](SWE_SMITH_AGENTIC_64_STATUS.md): task,
   q-first qualification, model smoke, failed RL smoke, and remaining gates.
2. [`runs/swesmith_online_status_v01/summary.json`](runs/swesmith_online_status_v01/summary.json):
   timestamped compact counts and failure status.
3. [`RESULTS.md`](RESULTS.md) and [`GPT_CONTEXT.md`](GPT_CONTEXT.md): canonical
   evidence table and claim/non-claim boundary.
4. For implementation review only: `scripts/swesmith_q_qualifier_worker.py`,
   `scripts/swesmith_agent_worker.py`, `scripts/swesmith_modal_executor.py`,
   and `scripts/train_swesmith_agent_grpo.py`.

Do not open raw task JSONL, local logs, caches, or model files first. They are
omitted from this repository. The small q-first pool manifests preserve the
pinned source revision and task-row hashes for regeneration.

## Decision-relevant delta

The original 64 official rows were only candidates; the reference-semantic
proxy requires a separate q-first screen. The first screen qualified 4/311.
At a timestamped *partial* snapshot of the expanded 724-candidate pool,
28/171 checked rows were q-valid; this is not a finalized selection or a
policy success rate. The first four q-valid tasks passed official bug-injection
self-consistency, all from one repository.

The pinned Qwen2.5-Coder-1.5B-Instruct weight passed full-hash verification and
an offline RTX 5090 load/generation smoke. A restricted, multi-edit agent on
real repositories then attempted a four-task, one-update GRPO smoke. It failed
in terminal q scoring with `callable_import_or_execution_failed` **before any
optimizer update**. There is no three-arm RL result, held-out solve rate, or
evidence that q improves learning. This agent replaces one function/method and
gets public-test feedback; it is not a full SWE-agent tool workflow.

The old Function-SWE trainer is a surrogate and must not be interpreted as the
real-task launcher. The previous exact-25 trajectory replay result, including
seven invalid endpoints, is unchanged.

## Reviewer questions

1. Is the q bank's signature/default/literal-based input distribution
   sufficiently independent of F2P/P2P public tests for the stated comparison?
2. What minimal fix to terminal q scoring preserves the frozen qualification
   definition and fails closed on import/execution errors?
3. After that fix, what one-update smoke would establish engineering readiness
   before freezing 40 q- and gold-qualified tasks and launching the three arms?
