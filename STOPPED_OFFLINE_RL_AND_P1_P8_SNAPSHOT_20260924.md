# Stopped offline LoRA and P1–P8 snapshot (2026-09-24)

This is an interrupted, descriptive snapshot, not a completed long-horizon RL experiment.

## Offline LoRA result

The frozen one-edit census supplied 448 P1 candidates across 28 tasks. The
20 train tasks contributed 320 fixed candidates; 8 held-out tasks contributed
128 fixed candidates. The Test arm used `Y + 0.5 p`; the Semantic arm used
`Y + 0.5 q`. Each rank-16 LoRA arm completed 20 offline updates with the same
1.5B base model. All 20 updates in each arm had a nonzero gradient. Reward
advantages were nonzero for 20/20 Test and 19/20 Semantic task groups.

The held-out check ranked the **same 128 frozen completions** by policy log
probability. It did not generate new repairs or measure a solve rate.

| Policy | Concordance with terminal Y (3 informative tasks) | Test reward (8) | Semantic reward (8) |
|---|---:|---:|---:|
| Base | 0.6581 | 0.6919 | 0.8018 |
| Test LoRA, update 20 | 0.6581 | 0.6887 | 0.8018 |
| Semantic LoRA, update 20 | 0.6581 | 0.6887 | 0.8018 |

These are task-macro pairwise concordances. The two trained arms did **not**
separate on this held-out ranking check. Nonzero gradients and changed LoRA
weights establish that optimization ran, not that repair ability improved.
This fixed-data procedure was not on-policy GRPO, did not test eight-step
credit assignment, and does not support an RL benefit claim for `q`.

Raw metrics and held-out ranking are in
[`evidence/stopped_snapshot_20260924/offline_448_lora_test_vs_semantic_v01/`](evidence/stopped_snapshot_20260924/offline_448_lora_test_vs_semantic_v01/).
The training and evaluation implementations are
[`train_swesmith_offline_448_lora.py`](scripts/train_swesmith_offline_448_lora.py)
and [`evaluate_swesmith_offline_448_lora.py`](scripts/evaluate_swesmith_offline_448_lora.py).
The 1.5B/LoRA configuration is
[`swesmith_offline_448_lora_v01.json`](configs/swesmith_offline_448_lora_v01.json).
The model adapters remain on the stopped server and are **not** part of this
GitHub snapshot.

## Interrupted P1–P8 capture

The separate capture-only process reused existing P1 candidates and generated
subsequent edits without running tests or `q` at each step. After the user's
stop instruction, its preserved JSONL contained **438 state records** from
**4 tasks**: P0: 4, P1: 55, P2: 55, P3–P8: 54 each. Thus **54 trajectories
reached P8**; one more had reached P2. The planned 448 eight-step trajectories
were **not** completed. The saved records contain source, function body,
source hash, and generated action metadata. P2–P8 have **no public-test
scores, no terminal outcomes, and no `q` values**. The offline `q` job was
stopped before grading. P1 scores exist only in the separate one-edit census;
they are not scores for this eight-step run.

The unmodified capture receipt still says `running` because the process was
terminated externally. It must not be interpreted as a live process or a
completed run. The exact stopped data are
[`states.jsonl`](evidence/stopped_snapshot_20260924/body_trajectories_capture_only_1p5b_28x16x8_v01/states.jsonl)
and [`run.json`](evidence/stopped_snapshot_20260924/body_trajectories_capture_only_1p5b_28x16x8_v01/run.json).
Their transfer archive had SHA-256
`7e03d4208d6384df05b0db960138808db9f177ce03c221e32744e4f300c1a156`.
The capture implementation is
[`capture_swesmith_body_trajectories.py`](scripts/capture_swesmith_body_trajectories.py).

No analysis of `q(P1),…,q(P8)` or trained-policy eight-step performance is
possible from these ungraded partial trajectories.
