# GPT review context

## Research question

The sole primary question is whether a simple program-semantic intermediate
state exists on real coding-agent trajectories and predicts terminal success
beyond sparse returns and public-test observations. The project must answer that
before critic scaling or online RL.

## Formal status

- `PANEL_FROZEN`: pass — 40 trajectories, 20 resolved and 20 unresolved.
- `TASK_BINDING`: partial — 25 exact current-task bindings; 15 profile fallbacks.
- `PATCH_FIELD_INTEGRITY`: fail — top-level patches are frequently empty or
  path-disjoint from the recorded editor calls.
- `STRUCTURED_REPLAY_RUNTIME`: qualified on one exact resolved task under
  udocker/PRoot.
- `PRoot_ENDPOINT_REPLAY_PASS`: pass — 5/5 Moto tasks reproduce target failure
  and agree with their declared endpoint.
- `MOTO_INTERMEDIATE_AUDIT`: complete — 52 states, 47 successful mutations, one
  negative transition, and one trajectory-induced invalid region.
- `EXACT25_BATCH`: complete — 25/25 summaries, 184 states.
- `EXACT25_STRICT_ENDPOINT_GATE`: partial — 18/25; seven invalid endpoints
  retained for runner/environment diagnosis.
- `INTERMEDIATE_SIGNAL_GATE`: first signal present, prevalence not estimated.
- `CRITIC_COMPARISON`: not run.
- `LIVE_AGENT_CONTINUATION`: not run.
- `ONLINE_RL`: not run.
- `REAL_SWE_SMITH_64_SELECTION`: candidate slice frozen; 64 official task rows,
  eight images, 48/16 split; not runtime-qualified.
- `REAL_MODAL_IMAGE_SMOKE`: one h11 image starts at `/testbed`.
- `REAL_MODAL_GOLD_SMOKE`: one h11 target test passes clean, fails after official
  bug injection, and passes again after reversal; not full grading.
- `Q_FIRST_QUALIFICATION`: first screen 4/311; expanded screen 28/171 at a
  timestamped partial snapshot, not a finalized task set.
- `PINNED_MODEL_GPU_SMOKE`: passed 16-token offline generation on RTX 5090.
- `RESTRICTED_AGENT_RL_SMOKE`: failed in terminal q scoring before any optimizer
  update; no completed GRPO update or formal three-arm result. This is a
  function/method replacement agent, not a full SWE-agent workflow.
- `AGENTIC_RL_ROLLOUT_OR_UPDATE`: no effective update. The Function-SWE trainer
  is a separate surrogate and cannot train on the real-task slice.
- `TWO_ARM_SURVIVAL_ATTEMPT`: one semantic optimizer step was executed, but
  8/8 edits were invalid, within-group rewards tied, and gradient was zero.
  Stopped without an effective parameter update, test-arm update, or heldout
  result; see `runs/swesmith_survival_status_v01/summary.json`.
- `BODY_CENSUS_28X16`: running; a frozen six-task snapshot shows 79/96
  executable candidates, task-level `q` variance on 6/6 versus public-test
  variance on 2/6, and 197/408 within-task public-test ties split by `q`.
  See `BODY_CENSUS_PARTIAL_6_TASKS_20260923.md`. This is observational
  information, not long-horizon credit or RL improvement.
- `EIGHT_STEP_TRAJECTORIES`: dependent continuation queued, not complete.
  It reuses each census P1, saves P0 through P8, and uses only public-test
  feedback online. `q(P0)`, `q(P4)`, and `q(P8)` are measured offline after
  all trajectories finish. The policy never sees q.

## Variants and evidence strata

1. **Exact bindings (primary replay stratum):** the selected `instance_id` exists
   in the pinned `SWE-bench/SWE-smith-py` revision.
2. **Profile fallbacks (sensitivity only):** the exact task is absent, but its
   repo profile and image remain identifiable. Do not combine these with exact
   tasks in a primary estimate.
3. **Structured replay:** reconstruct `create`, `insert`, and `str_replace`
   actions from `messages[].tool_calls`; require unique replacements and reject
   paths outside `/testbed`.
4. **Top-level patch:** retained only as an audited upstream field. It is not a
   trusted replay source.

## Claim boundary

Supported now:

- the panel and binding manifests are reproducibly frozen;
- a broad path-level integrity problem exists in this selected panel;
- 18/25 exact tasks meet the strict initialized, valid-endpoint replay gate
  under PRoot; seven remain invalid and are retained for diagnosis;
- the complete five-task Moto exact slice has endpoint self-consistency;
- terminal labels hide a true regression excursion and a recoverable invalid
  region in this slice.
- the wider exact-bound batch contains a second valid negative transition in
  SQLFluff, without establishing prevalence or future-value information.

Not supported now:

- prevalence or causal value of a certificate intermediate;
- resolved-versus-unresolved mechanistic conclusions from the selected counts;
- critic, sample-efficiency, decomposition, PPO, or general coding-agent gains.

The panel was selected with at least two structured edits, so its action/edit
statistics are selection-conditioned and are not population estimates.

## Evidence routing

- Canonical numbers and verdicts: [`RESULTS.md`](RESULTS.md)
- Pipeline and invariants: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Selection provenance: `runs/selection_v01/selection_manifest.json`
- Binding provenance: `runs/selection_v01/task_bindings_manifest.json`
- Patch integrity: `runs/patch_alignment_v01/patch_alignment_summary.json`
- First replay receipt: `runs/qualification_v01/replay_receipt.json`
- Moto aggregate: `runs/moto_intermediate_v01/aggregate_summary.json`
- Exact-25 aggregate and next decision: [`EXACT25_RESULTS_AND_NEXT_v01.md`](EXACT25_RESULTS_AND_NEXT_v01.md)
- Exact-25 per-task curves: `runs/exact25_intermediate_v01/aggregate_summary.json`
- Raw trajectory rows: omitted from the public repository; regenerate with the
  pinned selector and verify against the manifest hash.
- New real-task candidate manifest and claim boundary:
  [`SWE_SMITH_AGENTIC_64_STATUS.md`](SWE_SMITH_AGENTIC_64_STATUS.md) and
  `runs/swesmith_agentic_64_candidates_v01/manifest.json`. The compact
  `candidate_index.json` lists all 64 IDs; raw task rows are locally frozen
  and omitted from this public review repository.
- Timestamped q-first, model, and RL-smoke status:
  [`runs/swesmith_online_status_v01/summary.json`](runs/swesmith_online_status_v01/summary.json).

## Exact code symbols

- Panel construction: `select_swesmith_tool_trajectories.select_panel`
- Tool-boundary extraction: `select_swesmith_tool_trajectories.extract_structured_edits`
- Task binding: `fetch_swesmith_task_metadata.main`
- Patch audit: `audit_swesmith_patch_alignment.audit_row`
- Fail-closed replay: `replay_structured_edits.apply_call`
- Tool-outcome filtering: `replay_structured_edits.mutation_events`
- Edit-level evaluator: `run_moto_intermediate.evaluate_state`
- Moto aggregation: `summarize_moto_intermediate.main`
- Exact-25 aggregation: `summarize_exact25_intermediate.aggregate_summaries`
- Path confinement: `replay_structured_edits.confined_path`
- Real-task selection: `freeze_swesmith_agentic_candidates.eligible` and
  `freeze_swesmith_agentic_candidates.select`
- Public task index: `summarize_swesmith_agentic_candidates.summarize`
- One-edit census: `sample_swesmith_body_28x16.main` and `information_metrics`
- Body assembly and saved state: `swesmith_agent_edit.replace_callable_body`,
  `swesmith_agent_edit.current_callable_body`
- Public-only sequential rollout: `continue_swesmith_body_trajectories.main`
- Offline q: `grade_swesmith_trajectory_q.main`, worker `run(mode="proxy_only")`
- Eight-step observation: ready 16-sample P1 blocks stream to P8 with public
  feedback only; after completion, offline q covers every P0–P8 state, 3612
  observations before per-task source-hash deduplication. No RL update here.
- Real Modal checks: `smoke_swesmith_modal_image.main`,
  `smoke_swesmith_gold_modal.main`, `swesmith_gold_worker.main`

## Reviewer questions

1. Does the patch-integrity evidence justify making structured tool calls the
   sole replay source, or is another upstream field needed?
2. Is the exact-binding/profile-fallback separation strict enough?
3. What is the smallest pre-registered intermediate-state statistic that can be
   evaluated on a balanced exact-bound replay subset without training a critic?
4. Should invalid collection regions be modeled as a distinct state, or as a
   censored observation outside the two-dimensional `(F,R)` phenotype?
5. Is the proposed matched-budget live-agent continuation comparison the
   smallest decisive test of future-value information after replay repair?
