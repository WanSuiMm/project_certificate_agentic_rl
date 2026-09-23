# SWE-smith agentic RL task and runtime status

Date: 2026-09-23. This is a new, user-directed pivot to real repository tasks,
not an amendment that makes the earlier Function-SWE result into agentic RL.
No effective RL parameter update or held-out comparison has completed. An
early restricted engineering smoke failed in the terminal q scorer before an
optimizer update.
Later one-edit runs reached an optimizer step but had zero gradients; no
effective RL update or held-out comparison exists. See
`runs/swesmith_survival_status_v01/summary.json` for the later attempt.

## Official task source and 64-task candidate slice

- Source: `SWE-bench/SWE-smith-py`, pinned revision
  `77cab9055d42ab4a5c25c89a8f937096db13558e`.
- Script: `scripts/freeze_swesmith_agentic_candidates.py`.
- Full local artifact: `runs/swesmith_agentic_64_candidates_v01/tasks.jsonl`, SHA-256
  `475b0731feed69a2bccd97d8635c26fe42a389f574c419757d892e87e3eb2013`;
  selection details in the adjacent `manifest.json`. The public repository
  contains `candidate_index.json` (IDs, images, splits and test counts), but
  omits the raw issue text and patches. The pinned selector regenerates the
  full artifact, which must match the manifest hash before use.
- Scanned 50,908 official task rows. Deterministic, outcome-blind filters yielded
  4,510 candidates and 52 images with at least eight candidates. Eight images
  were selected by seeded hash rank; eight tasks per image, split 6 train and
  2 held-out per image (48/16). Each row preserves the official instance ID,
  issue, bug patch, image name, FAIL_TO_PASS and PASS_TO_PASS lists.
- This is a **candidate** slice, not 64 qualified runtime tasks. Each image and
  task still needs gold self-consistency testing before formal RL use. Neither
  hidden function cases nor reference-probe inputs are present in SWE-smith.

## Real Modal environment check

`scripts/smoke_swesmith_modal_image.py` pulled the official public h11 image
`swebench/swesmith.x86_64.python-hyper_1776_h11.bed0dd4a` into a real Modal
Sandbox. It opened `/testbed` and exited successfully. The h11 instance
`python-hyper__h11.bed0dd4a.func_basic__430lolni` then passed the bounded
gold check in `scripts/smoke_swesmith_gold_modal.py`:

```text
clean image, target test: pass
official bug injection: target test fails
reverse official bug patch: target test passes again
```

The first test attempt used the image's bare base Python and was invalid because
pytest lives in `/opt/miniconda3/envs/testbed`. The corrected attempt used that
interpreter and passed; its machine-specific operational receipt is retained
locally and omitted from the public repository.
This verifies one target test of one task, not all 64 tasks or full official
grading. Candidate code was executed only inside a network-blocked Modal Sandbox.

## Actual agentic RL boundary

A valid agentic rollout must give the model a sequence of repo observations and
actions (inspect files, edit files, run public commands) in one isolated task
environment, produce a final patch, and grade that patch in a separate fresh
environment. GRPO then updates the probabilities of the model's action tokens
using a trajectory-level reward. Merely sampling replacement functions is not
the same experiment. The existing `train_oracle_proxy_grpo.py` implements the
earlier Function-SWE surrogate and **must not** be launched on this task slice.

The real-task reward comparison still requires an operational definition of
`q(P)` on full repository patches, plus a matched public-test score `p_T` and
terminal solve label `Y`. SWE-smith supplies F2P/P2P test identifiers and
images but no automatically frozen 256-input reference probe bank. Reusing
F2P/P2P as `q` would make it a test reward, not the semantic proxy under study.

The next engineering gates are: check gold self-consistency across candidate
images/tasks; wire a real multi-turn agent harness to Modal; construct and freeze
the reference-behavior bank independently of public tests; run a matched
one-update 5090 smoke; only then dispatch the three RL arms. Preserve all
failures in the denominator or make exclusions before any policy outcomes.

## Q-first and fixed-model update (2026-09-23)

The original 64 are not assumed q-compatible. A new outcome-blind selector
streams the same pinned 50,908-row dataset, prioritizes function-level tasks,
and qualifies each in its official network-blocked Modal image. The 256-input
bank is generated from callable signatures, defaults, source literals and
generic relational perturbations, never F2P/P2P inputs. It is kept only when
the clean reference is deterministic, the official bug changes 13–243 of the
256 observations, and the callable signature remains unchanged. The first
screen qualified 4/311 tasks; its low yield is retained, not hidden. A broader
pool of 724 candidates across 15 images is currently being screened. These
are qualification counts, not learning results. A separate gold check found
the first four qualified tasks reproduced the official failure.

The pinned Qwen2.5-Coder-1.5B-Instruct weight was downloaded in eight parallel
range streams and verified by full SHA-256 against the pinned Hugging Face blob.
An offline RTX 5090 BF16 load/generation smoke passed: 16 new tokens and
2.887 GiB peak allocated memory. A four-task, one-update engineering smoke
was attempted using three sequential replacement edits and official public
test feedback per rollout, with terminal `q` hidden from the agent. This is a
restricted function/method-edit agent on real repositories, **not** a full
SWE-agent shell/editor workflow. It failed in the terminal q scorer with
`worker_error/SkipTask/callable_import_or_execution_failed` before any optimizer
update. This is an engineering failure, not a policy outcome. Formal arms still
require a completed nonzero-gradient one-update smoke. The later exploratory
20/8 slice was frozen without full gold qualification and stopped after its
first zero-gradient semantic update; it is not a completed policy comparison.
