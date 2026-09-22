# Certificate-Supervised Agentic Value Learning

This project asks whether program-semantic certificates expose a useful
intermediate state for predicting terminal success on real coding-agent
trajectories. The current phase is a frozen SWE-smith trajectory audit, not
critic training or online RL.

## Start here

1. [`GPT_CONTEXT.md`](GPT_CONTEXT.md): compact task definition, claim boundary,
   code map, and reviewer questions.
2. [`RESULTS.md`](RESULTS.md): canonical aggregates and formal status.
3. [`ARCHITECTURE.md`](ARCHITECTURE.md): data flow, invariants, and exact scripts.
4. [`MOTO_INTERMEDIATE_PROTOCOL_v01.md`](MOTO_INTERMEDIATE_PROTOCOL_v01.md):
   the frozen edit-level replay protocol.
5. [`PROJECT.md`](PROJECT.md): broader research contract and next gate.

The two large raw JSONL files are frozen locally but intentionally omitted from
the public review repository because they contain third-party issue text and
machine-specific paths. Their hashes remain in the manifests, and the pinned
selector/binder scripts regenerate them.

## Current result

- Frozen panel: 40 trajectories, balanced 20 resolved / 20 unresolved, from 24
  repositories; all use `claude-3-5-sonnet-20241022`.
- Current task binding: 25 exact tasks and 15 repo-profile fallbacks.
- Integrity finding: only 1 of 13 resolved trajectories with a non-create editor
  mutation has any path overlap between its actual editor calls and the dataset's
  top-level `patch`. Empty or disjoint patches are not valid replay inputs.
- Runtime qualification: one exact Conan task changes from `1 failed, 1 passed`
  after task initialization to `2 passed` after fail-closed structured-edit
  replay, agreeing with `resolved=true`.
- Moto exact slice: all five endpoint replays agree with their declared outcome
  (`PRoot_ENDPOINT_REPLAY_PASS=true`) across 52 recorded states and 47 successful
  filesystem mutations.
- Intermediate signal: one unresolved trajectory moves `(F,R)` from `(4,0)` to
  `(4,12)` and then `(4,1)`; one resolved trajectory reaches `(0,0)`, becomes
  temporarily uncollectable after a bad import edit, and later returns to
  `(0,0)`.
- Scientific status: the slice establishes that endpoint labels hide meaningful
  edit-level path structure. It does not estimate prevalence or show a
  certificate-trained critic or online-RL gain.

## Reproduce compact checks

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/audit_prior.py
python scripts/select_swesmith_tool_trajectories.py --output /tmp/selection_v01
python scripts/audit_swesmith_patch_alignment.py \
  --trajectories /tmp/selection_v01/selected_trajectories.jsonl \
  --output-dir /tmp/patch_alignment_check
```

The replay requires the pinned SWE-smith image. PRoot is the current experimental
execution substrate; each task is initialized from a fresh image-derived rootfs.
