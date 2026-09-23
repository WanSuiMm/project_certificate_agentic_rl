# GPT handoff: one-edit RL survival attempt stopped

- Review base: `5439d82f88b79f186cf93b6c91c28b44ac01fe7c`
- Evidence head: `ce38de1253538a56c42fb5dc2494679ab1400f1c`
- This handoff is metadata-only; the historical exact-25 replay claims are
  unchanged. Review only the evidence-head delta.

## Read first

1. [`runs/swesmith_survival_status_v01/summary.json`](runs/swesmith_survival_status_v01/summary.json):
   compact selection, launch, failure, and non-claim record.
2. [`RESULTS.md`](RESULTS.md) and [`PROJECT.md`](PROJECT.md): current status and
   research boundary.
3. For implementation review: `scripts/train_swesmith_agent_grpo.py`,
   `scripts/swesmith_agent_edit.py`, `scripts/swesmith_agent_worker.py`, and
   `configs/swesmith_survival_20x8_v01.json`.

Do not open raw task rows, model caches, server logs, or local launch receipts;
they are intentionally absent from GitHub.

## Decision-relevant delta

The expanded q-first screen stopped at 29/636 q-valid tasks across three
images. A 20-train / 8-heldout set was frozen, but the third image contributes
only one task. A one-edit Test-vs-Semantic GRPO comparison was attempted, not
completed. Its first semantic update produced 8/8 invalid edits, tied rewards
within both groups, and `grad_norm=0`; the process was stopped. Subsequent
one-task smokes also had zero gradients. There is no effective policy update,
test-arm update, held-out solve rate, or evidence that q improves RL.

The code now scores candidate import failures as `q=0` and candidate public
test collection failures as `p_T=0, solved=false`, while retaining fail-closed
infrastructure errors. Non-semantic arms do not request q. Output generation
has no independent token cap, ending at EOS or the pinned model's 32,768-token
context boundary. These are engineering changes, not positive scientific
results. The earlier Function-SWE trainer remains a separate surrogate.

## Reviewer questions

1. Does this 1.5B replacement-function action format produce enough valid,
   reward-distinct rollouts to justify another RL attempt?
2. If not, should the one-edit survival line stop here rather than expanding
   the task pool or inventing more proxy variants?
