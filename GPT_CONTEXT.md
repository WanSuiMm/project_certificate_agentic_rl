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
- `INTERMEDIATE_SIGNAL_GATE`: first signal present, prevalence not estimated.
- `CRITIC_COMPARISON`: not run.
- `ONLINE_RL`: disabled.

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
- one exact task was initialized and replayed successfully under PRoot.
- the complete five-task Moto exact slice has endpoint self-consistency;
- terminal labels hide a true regression excursion and a recoverable invalid
  region in this slice.

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
- Raw trajectory rows: omitted from the public repository; regenerate with the
  pinned selector and verify against the manifest hash.

## Exact code symbols

- Panel construction: `select_swesmith_tool_trajectories.select_panel`
- Tool-boundary extraction: `select_swesmith_tool_trajectories.extract_structured_edits`
- Task binding: `fetch_swesmith_task_metadata.main`
- Patch audit: `audit_swesmith_patch_alignment.audit_row`
- Fail-closed replay: `replay_structured_edits.apply_call`
- Tool-outcome filtering: `replay_structured_edits.mutation_events`
- Edit-level evaluator: `run_moto_intermediate.evaluate_state`
- Moto aggregation: `summarize_moto_intermediate.main`
- Path confinement: `replay_structured_edits.confined_path`

## Reviewer questions

1. Does the patch-integrity evidence justify making structured tool calls the
   sole replay source, or is another upstream field needed?
2. Is the exact-binding/profile-fallback separation strict enough?
3. What is the smallest pre-registered intermediate-state statistic that can be
   evaluated on a balanced exact-bound replay subset without training a critic?
4. Should invalid collection regions be modeled as a distinct state, or as a
   censored observation outside the two-dimensional `(F,R)` phenotype?
