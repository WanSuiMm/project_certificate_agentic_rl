# Function-SWE RL runtime

Status: code and trusted-fixture smoke only. No 64-task manifest, isolated
executor, GPU training, or held-out result exists yet.

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

Once those prerequisites exist, the command shape for each arm is:

```text
python scripts/train_oracle_proxy_grpo.py --arm semantic \
  --config configs/oracle_proxy_grpo_survival_v01.json \
  --manifest MANIFEST_PATH --output-dir NEW_RUN_DIR \
  --executor-command-json '["ISOLATED_SCORER_COMMAND", "ARG"]' \
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
official full-repository SWE-smith solve. No AutoDL or Modal deployment wrapper
has been implemented or validated.
