# Oracle Credit Benchmark v01

Question: within a task's frozen P1 candidates, does reference agreement q
predict continuation success beyond the public-test fraction p?

This is frozen-policy evaluation. There is no optimizer or policy update.
The completed pilot and claim boundary are in
[`ORACLE_CREDIT_RESULTS_20260925.md`](ORACLE_CREDIT_RESULTS_20260925.md).
The three training experiments in `LONG_HORIZON_GRPO_PROTOCOL_20260924.md`
remain implemented but are not the active execution plan.

## Frozen pilot

- First six task IDs in the existing 28-task frozen selection, retaining all
  16 original candidates per task; no selection by future outcomes.
- Four independently sampled continuations per candidate, P1 -> P2 -> ... ->
  P8, with public tests after every edit and final selected-test resolution Y8.
- 96 P1 candidates, 384 endpoints, 2,688 new edits. A solved intermediate
  state still receives the remaining edits; the target is fixed-horizon Y8.
- Qwen2.5-Coder-1.5B-Instruct at the pinned model revision, frozen weights,
  1,024 tokens per edit. Decoding matches the previous generator's effective
  settings: temperature 1, top-p 1, top-k 20, repetition penalty 1.1.
- The policy sees issue, signature/current function body and the preceding
  public feedback. It sees neither reference source, probe inputs nor q.
- q(P1) comes from the frozen census. Syntax-invalid P1 edits are no-ops:
  restore the actual P0 public/q measurements instead of its old penalty zeros.
  Unmeasured/invalid q remains null, with missing-proxy coverage reported.
- Each P1 source must exactly match its saved SHA-256. Existing valid public
  measurements must reproduce before continuing. Invalid actions later keep
  the source and expose invalid-edit feedback; their resulting state is tested.

Q_i = mean_k Y8_ik estimates V(pi, h1_i), and hence Q(pi, h0, a1_i) under
the fixed initial transition and gamma=1. It measures this specific frozen
policy, feedback interface and remaining seven-edit budget. Y8 is the official
selected public-test gate, not an independent hidden-correctness oracle.

## Measurements and interpretation

Compare task-level Spearman(p,Q) and Spearman(q,Q), and pairwise directional
accuracy (proxy ties count 0.5) on empirically non-tied Q pairs. The primary
subset contains two P1-unsolved candidates with identical p. Report proxy
coverage, empirical oracle ties, Wilson 95% intervals, and a conservative
directional subset whose intervals do not overlap. Pair counts are descriptive;
tasks, not dependent candidate pairs, are the aggregation unit.

K=4 is a resolution screen. An all-zero/constant observed Q or insufficient
confident comparisons is INCONCLUSIVE, not a negative semantic-proxy result.
There is no automatic KEEP/KILL from this small pilot. A negative result is
scoped to direct q ranking under this policy and budget. PRM and RL are deferred.

## Execution

```bash
python scripts/run_swesmith_oracle_credit.py \
  --config configs/swesmith_oracle_credit_6x16x4_v01.json \
  --selection FROZEN_SELECTION.json --census CENSUS/results.jsonl \
  --tasks TASKS.jsonl --q-results Q_QUALIFICATION/results.jsonl \
  --output-dir NEW_RUN_DIRECTORY

python scripts/summarize_swesmith_oracle_credit.py --run-dir RUN_DIRECTORY
```

Use the same inputs and output directory to resume. The runner pins input and
source-code hashes, journals each generated batch atomically, and saves every
scored state and final endpoint. It rejects changed inputs and broken feedback
chains. `run.json` contains progress, configuration and provenance;
`candidates.jsonl`, `action_batches/`, `states.jsonl`, `outcomes.jsonl` preserve
the actual data. A complete run automatically writes `summary.json`.

When recovery requires changed runner code, keep the failed or explicitly
interrupted run untouched and start a new run with
`--import-run-dir PRIOR_RUN_DIRECTORY`. The importer checks
the frozen inputs, copies the action/state/outcome journal, records source hashes
in `import_manifest.json`, and resumes only missing scores/actions. A persistent
worker that ends before replying triggers one stateless retry of that exact
source; it never triggers resampling or synthetic success.

The buffered continuation runner advances any trajectory as soon as its public
feedback arrives. It batches ready actions for GPU generation and journals the
whole batch before scoring. A slow public test cannot block other trajectories
at the same step. Future actions still require their own preceding public-test
observation; no open-loop or q-conditioned generation is introduced.

Performance choices: frozen inference with KV cache and SDPA; batches of eight
actions; four persistent task sandboxes scoring concurrently while the GPU
generates the next batch. Each scorer receives the complete target source;
fresh Python subprocesses avoid interpreter-state carryover. Bytecode writing
is disabled; unexpected source/repository mutation aborts the run. No q grader
is called on P2–P8. Runtime and costs must be measured on the deployed system.
