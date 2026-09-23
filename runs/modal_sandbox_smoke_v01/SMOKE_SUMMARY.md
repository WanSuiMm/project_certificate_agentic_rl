# Modal scorer live smoke (2026-09-23)

Verdict: **PASS**, engineering connectivity only; no RL update or held-out task
was run. In two fresh Modal Sandboxes, the trusted toy buggy function scored
`solved=false, q=0`, and its repair scored `solved=true, p_T=1, q=1`.

The client ran on the existing 5090 host using Modal Python SDK 1.5.5. Each
Sandbox had outbound network blocked, no mounted secrets or volumes, a
45-second lifetime, and CPU/memory limits. The receipt was saved outside the
repository. The sandbox code hashes in that receipt correspond to
`scripts/function_swe.py`, `scripts/function_swe_worker.py`, and
`scripts/modal_function_swe_executor.py` at commit `9bb1fd0`.

This qualifies the scorer transport on a trusted toy fixture only. It does not
qualify a 64-task Function-SWE manifest, demonstrate model training, measure
full-scale latency/cost, or show a scientific reward advantage. At the check,
neither 5090 had the 24 GiB free VRAM required by the formal trainer gate.
