# One-edit body-action census: six-task provisional snapshot

This is a fixed **partial snapshot**, not the 28-task endpoint. At 2026-09-23
15:15 UTC, the first six of 28 frozen SWE-smith tasks had all 16 sampled
one-edit candidates recorded (96 candidates). The remaining tasks were still
running. Four of these six tasks are from `stanfordnlp/string2string`, one from
`luozhouyang/python-string-similarity`, and one from `tobymao/sqlglot`.

The policy is pinned `Qwen/Qwen2.5-Coder-1.5B-Instruct` revision
`d18ecad881a290f0c792f1c2b5dbee1820d4a424`, with a single replacement-body
action and a 1024-token generation cap. The 28-task selection SHA-256 is
`7b8c94a6f81d48be6ac2c710d2edd992712a80318a8ce523e868e3bdf1846080`.
Each executable candidate was evaluated with public tests and the same frozen
256-input reference-agreement bank. No policy update was performed.

| Provisional measure on the six complete tasks | Observation |
| --- | ---: |
| Body actions assembled into valid Python | 81 / 96 |
| Candidates with valid public-test observation | 79 / 96 |
| Tasks with more than one public-test pass fraction among executable candidates | 2 / 6 |
| Tasks with more than one reference-agreement `q` among executable candidates | 6 / 6 |
| Within-task pairs with equal public-test pass fraction | 408 |
| Such pairs with unequal `q` | 197 / 408 |
| Unresolved, equal-public-test pairs with unequal `q` | 197 / 405 |
| Public-test-resolved candidates | 3 / 79, all from one task |

These observations show **additional discrimination**, not useful temporal
credit or improved RL. Pair counts are descriptive within-task comparisons,
not 408 independent statistical units. A candidate can be unresolved even
when `q=1` on the finite probe bank, so `q` is not a complete correctness
oracle. The former 1/4 structural-validity diagnosis was a body-indentation
assembler bug: all four frozen outputs parsed after the minimal fix; this
snapshot uses the fixed assembler.

The follow-on observational run will continue these same frozen first edits
to eight sequential edits per trajectory. It will save each target-file state
and public-test feedback without online `q`; only after every trajectory
reaches `P8` will the separate grader measure `q(P0)`, `q(P4)`, and `q(P8)`.
Neither that run nor any long-horizon RL comparison has a result yet.
