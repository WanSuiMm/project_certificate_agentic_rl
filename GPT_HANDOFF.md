# GPT handoff: stopped offline LoRA and partial P1–P8 capture

- Review base: `36165fea8b77414ca97a41025184cfcb1ce9b0fa`
- Evidence head: `5737257` (commit containing code, summary, and raw snapshot)
- This handoff is metadata-only; review the evidence-head delta first.

## Read first

1. [`STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md`](STOPPED_OFFLINE_RL_AND_P1_P8_SNAPSHOT_20260924.md): exact outcomes and claim limits.
2. [`RESULTS.md`](RESULTS.md): canonical status table.
3. [`configs/swesmith_offline_448_lora_v01.json`](configs/swesmith_offline_448_lora_v01.json) and the new `scripts/` files: implementation.

The 9 MB partial source-state JSONL is secondary evidence; do not open it
first. Earlier Moto/exact-25 replay claims are unchanged.

## Decision-relevant delta

Fixed-data Test and Semantic LoRA arms each completed 20 updates with nonzero
gradients. On the same 128 held-out frozen completions, their task-macro
candidate-ranking concordances were equal: 0.6581 for terminal outcome,
0.6887 for Test reward, and 0.8018 for Semantic reward. This is **not** a
solve-rate or on-policy RL result; it provides no evidence of a semantic-arm
advantage. There were no new held-out rollouts.

The requested capture-only eight-step run was terminated after 438 state rows:
54 complete P1–P8 trajectories across four tasks, far short of 448 planned.
P2–P8 contain source states only: **no test, terminal, or q values**. The
deferred offline q process did not start. The raw receipt still says `running`
because of external termination; the stopped-snapshot document governs status.

Validation: new scripts passed `py_compile`; `python -m pytest -q tests`
passed 51 tests. Whole-repository pytest collection collides on two unrelated
user-pasted `test_probe.py` copies; no project test failed.

## Reviewer questions

1. Does the fixed-candidate ranking have enough sensitivity to detect a useful policy change after only 20 offline updates?
2. What new on-policy, held-out solve-rate experiment would be necessary before claiming that semantic `q` improves RL?
3. Should the ungraded, incomplete P1–P8 state capture be retained solely as a reproducibility artifact?
