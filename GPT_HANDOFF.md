# GPT handoff: real SWE-smith agentic-RL preparation

- Review base: `4a1eca5ae3a1ec0ad806e2d37bfd42b70705f4e7`
- Evidence head: `fdfd157da0e129349cd6517fa232ddd4308881d5`
- This is a metadata-only handoff after the evidence head; do not reread the
  unchanged exact-25 raw curves or interpret this commit as new experiment data.

## Read only these first

1. [`SWE_SMITH_AGENTIC_64_STATUS.md`](SWE_SMITH_AGENTIC_64_STATUS.md): current
   real-task claim boundary and one-task Modal gold smoke.
2. [`runs/swesmith_agentic_64_candidates_v01/manifest.json`](runs/swesmith_agentic_64_candidates_v01/manifest.json):
   pinned dataset, deterministic filter, split and frozen full-row hash.
3. [`candidate_index.json`](runs/swesmith_agentic_64_candidates_v01/candidate_index.json):
   64 official IDs and their image/split/test counts.
4. Only if reviewing implementation: `scripts/freeze_swesmith_agentic_candidates.py`,
   `scripts/smoke_swesmith_modal_image.py`, `scripts/smoke_swesmith_gold_modal.py`,
   and `scripts/swesmith_gold_worker.py`.

The full raw task JSONL (issue text and patches), HF cache, and machine-specific
receipts are deliberately not published. The pinned selector regenerates the
full task rows; verify its SHA-256 against the manifest. Do not try to execute
the compact index as if it contained the patches.

## Decision-relevant delta

From the official pinned `SWE-bench/SWE-smith-py` revision, 50,908 rows were
scanned and 64 task candidates selected: eight images, eight tasks per image,
48 training and 16 held-out. This is selection, **not** qualification of all
64 environments. One official h11 image started in Modal; for one official
task, a target test passed clean, failed after bug injection and passed after
reversal. The first attempt used the wrong base interpreter and was invalid;
the passing check used SWE-smith's `testbed` Conda interpreter.

The historical exact-25 replay result and its seven invalid endpoints are
unchanged. No multi-turn agent rollout, repository-level semantic proxy, GPU
model inference, GRPO update, or held-out solve result exists. In particular,
`scripts/train_oracle_proxy_grpo.py` is the earlier Function-SWE surrogate and
must **not** be run on the new real-task slice.

## Reviewer questions

1. Is this outcome-blind, image-reuse-aware 48/16 candidate selection suitable
   as an engineering screen, with all later image/task exclusions recorded
   before any policy outcomes?
2. What is the smallest faithful full-repository reference-behavior bank for
   `q(P)` that is not merely a repackaging of the official F2P/P2P tests?
3. Which existing agent harness can be connected to Modal's persistent task
   Sandbox and a separate fresh grading Sandbox on one 32 GiB GPU, without
   representing the Function-SWE surrogate as agentic RL?
