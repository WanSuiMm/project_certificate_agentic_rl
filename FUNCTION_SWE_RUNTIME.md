# Function-SWE RL runtime

Status: trusted-fixture mock and real two-score Modal smoke passed on
2026-09-23, including after deployment to the new 5090 host. Remote unit tests
pass (36/36). No 64-task manifest, pinned-model GPU smoke, GPU training, or
held-out result exists yet. The large pinned weight download was stopped after
a bounded network check; no model inference was claimed.
See `runs/modal_sandbox_smoke_v01/SMOKE_SUMMARY.md` for the claim boundary.

`scripts/freeze_function_task.py` accepts one JSON spec with `task_id`, `issue`,
`target_function`, `buggy_source`, `reference_source`, `provenance`, nonempty
`public_cases` and `hidden_cases` (each case has `args`, `kwargs`, `expect`), and
exactly 256 disjoint `probe_inputs` (each has `args`, `kwargs`). An observation is
`{"kind":"return","value":...}` or `{"kind":"exception","type":"..."}`;
supported return values are finite JSON primitives/containers and tuples via
`{"__tuple__":[...]}`. The clean reference supplies **only** probe expectations;
the supplied public/hidden expectations must be checked against their original
upstream tests during curation. `provenance` should identify that source and its
hash. The 256 inputs, source hashes, cases and task IDs are frozen in each output.
Task qualification executes source code: run it inside an isolated sandbox.

`scripts/function_swe_manifest.py --task-dir TASK_DIR --output MANIFEST_PATH`
freezes exactly 64 task files, splitting 48/16 with seed zero. `TASK_DIR` must
be under the manifest's parent. The files are hash-checked on every load.

The controller never executes candidate code directly. Its
`--executor-command-json` is a JSON string array naming a separately isolated
command that accepts one JSON request on stdin and returns one JSON score on
stdout. `scripts/function_swe_worker.py` implements the scoring protocol, but
**running it with bare Python on the training host is unsafe**: candidate code
can import modules or access files. The caller must enforce actual process or
container isolation, resource limits, no secrets, and network/filesystem
restrictions. A JSON isolation receipt with `isolated_code_execution: true` and
`smoke_passed: true` is a preflight record, not proof of isolation by itself.
Worker faults abort training; they are never treated as zero reward.

The Modal adapter in `scripts/modal_function_swe_executor.py` creates a fresh
Sandbox for **each score call**, with outbound network blocked, no mounted
secrets/volumes, a 45-second lifetime, and CPU/memory limits. This isolates
consecutive candidate evaluations, but may be costly at full scale; measure
latency and cost on a bounded smoke before committing to 100 updates. Configure
Modal authentication on the training host outside the repository; never pass
its token to the scorer Sandbox or save it in a receipt. After installing
`requirements-rl.txt`, run the two-score live smoke:

```text
python scripts/smoke_modal_function_swe.py --receipt NEW_RECEIPT_PATH
```

It writes a receipt only after both a buggy and repaired trusted toy function
score correctly in real Modal Sandboxes. The trainer checks the receipt against
the current scorer and adapter source hashes. A passing receipt is necessary,
but task qualification and a one-update GPU smoke are still separate gates.

Once those prerequisites exist, the command shape for each arm is:

```text
python scripts/train_oracle_proxy_grpo.py --arm semantic --executor modal \
  --config configs/oracle_proxy_grpo_survival_v01.json \
  --manifest MANIFEST_PATH --output-dir NEW_RUN_DIR \
  --isolation-receipt RECEIPT_PATH --device cuda:0
```

Install `requirements-rl.txt` in the training environment. Repeat the command
with `--arm terminal` and `--arm test`, changing `NEW_RUN_DIR` each time.
Use `--smoke-updates 1` with a **new** output directory for a nonformal
end-to-end check. The three formal arms require distinct output directories and
start independently from the same pinned base model. The trainer records setup
hashes, task schedule, rollouts, training metrics and deterministic held-out
pass@1 at updates 0/20/40/60/80/100. Held-out execution omits proxy scoring.
The endpoint here is Function-SWE pure-function test solving; it is **not** an
official full-repository SWE-smith solve. The Modal adapter has a live toy
transport smoke but no full-scale performance measurement. No AutoDL deployment
wrapper exists.
