# Theory boundary, conditional story, and related work

This is a concise synthesis of three user-supplied discussion notes: a
September 24 theory recheck, a **conditional** paper storyline, and a proposed
oracle-credit reading map. It records their useful ideas without converting
proposals, examples, or literature analogies into results of this repository.
For the original derivations and numerical-check provenance, start with
[`THEORY_AND_HANDOFF_SUMMARY_20260925.md`](THEORY_AND_HANDOFF_SUMMARY_20260925.md).
For the latest SWE evidence, read
[`ORACLE_P2_POLICY_DYNAMICS_20260925.md`](ORACLE_P2_POLICY_DYNAMICS_20260925.md).

## 1. The theory recheck: three quantities, not one

1. **Analyzer imprecision** asks how far an approximate analysis is from the
   behavior it analyzes. The [POPL 2026 error-propagation logic](https://popl26.sigplan.org/details/POPL-2026-popl-research-papers/67/A-Logic-for-the-Imprecision-of-Abstract-Interpretations)
   addresses this kind of bound; it does not directly estimate how many more
   edits an LLM needs to solve a task.
2. **Current program error** can be defined under a fixed specification,
   input distribution, and loss as `E(P)=E_x[loss(P,x)]`. This project's frozen
   256-input reference agreement is a narrow empirical `q(P)=1-E(P)` when
   loss is reference-observation mismatch. `q(P')-q(P)` describes a change in
   *present program behavior* on that bank, not certified general correctness.
3. **Policy value/advantage** asks whether the *particular continuation policy*,
   given its history and remaining budget, can finish the task. Even an exact
   current-error measure does not include the policy's future edit dynamics.

A temporary semantic regression can be a useful refactor; reading or searching
can be valuable without changing the program at all. Consequently,
`Delta q_t` is not generally `A^pi(h_t,a_t)`. The supplied two-action
counterexample and independent 300-tree/4,880-trajectory numerical recheck
are algebraic/finite-world checks, **not** SWE or RL-training evidence.

For finite episodes with `gamma=1`, naively summing
`r'_t=r_t+alpha(q_{t+1}-q_t)` yields `Y+alpha(q_T-q_0)` and therefore changes
the optimized objective through terminal `q_T`. Ranking each solved sample
above each unsolved sample does not guarantee that expected solve-rate policy
ordering is preserved. Objective-preserving potential shaping requires the
correct terminal settlement. With a matched critic `Vhat=Phi+What`,

```text
r_t + gamma Phi(h_{t+1}) - Phi(h_t)
    + gamma What(h_{t+1}) - What(h_t)
  = r_t + gamma Vhat(h_{t+1}) - Vhat(h_t).
```

Thus the TD residuals and their [GAE](https://arxiv.org/abs/1506.02438)
sums are identical under this matched parameterization. Making reward numbers
look dense cannot, by itself, break equal-return whole-trajectory GRPO ties
while preserving the original objective. Potential shaping and Q-value
initialization have a known [relationship](https://arxiv.org/abs/1106.5267);
the algebraic factorization alone is not this project's novelty.

The remaining research question is empirical: **can an executable semantic
coordinate help a finite-data critic estimate future value and step-wise
advantage more accurately, then improve online learning under the same
terminal task reward?** The discussion note proposed capacity-matched
`V_test=f(h,p)` versus `V_sem=f(h,p,q)` critics, with both actors receiving
only normal public feedback. That comparison has **not** been run here.

## 2. The attractive paper story is a conditional evidence chain

The second note's central distinction is useful:

```text
Phi(P)       = measurable current artifact semantics
W^pi(h)      = future repairability under a policy, history, and budget
V^pi(h)     = Phi(P) + W^pi(h)   [algebraic representation]
```

Fixing `Phi` makes the residual `W=V-Phi` well-defined, but this identity
does **not** show `W` is easy to learn or that `Phi` improves policy learning.
The proposed storyline would require separate evidence:

| Link | Required observation | Current status |
|---|---|---|
| State discrimination | Same public feedback, distinct semantic states | Initial one-edit evidence only; scoped to its selected slice. |
| Future-value relevance | `q` predicts held-out continuation value beyond public tests | Not established; K=4 Oracle Credit and P2 drift results are noisy/mixed. |
| Critic mechanism | Same-capacity semantic critic improves held-out value/advantage estimation | Not run. |
| RL intervention | Same terminal reward and interaction budget; semantic critic improves held-out solve-rate learning | Not run. |
| Deployment extension | Reference-free executable constraints retain the gain | Not run; clean-reference `q` is privileged. |

Only if those links hold could one responsibly say that executable program
semantics help coding RL by making value easier to learn **without changing the
task reward**. A possible paper title or learning-curve sketch in the note is
an aspiration, not a reported result. The eight repeated edits in the current
restricted body-only agent also do not automatically constitute the full
tool-using, long-horizon dependence implied by that strongest story.

## 3. Related-work map, checked against primary sources

| Work | Relevant contribution | Boundary for this project |
|---|---|---|
| [Legibility is Not Interpretability](https://arxiv.org/abs/2609.04194) (2026) | Estimates reasoning-step importance via Monte Carlo changes in expected final reward, then tests how well judges and critics recover it. | CoT text, not coding artifacts; empirical MC advantage is noisy, not literal ground truth. A methodological neighbor for an oracle-credit benchmark. |
| [AgentPRM](https://arxiv.org/abs/2502.10325) (2025) | Uses MC rollout reward targets in an agent process-reward/actor–critic framework. | Shows the learned-process-value route; it is not evidence that static executable `q` approximates value. |
| [VPR](https://arxiv.org/abs/2605.10325) (2026) | Uses symbolic/algorithmic intermediate verification for turn-level process rewards in dynamic deduction, logical reasoning, and probabilistic inference. | Assumes reliable intermediate oracles; current code repair does not have a verified action-level correctness oracle. The supplied note's Sokoban/Sudoku/Minesweeper list is not the paper's stated three settings. |
| [SWE-Shepherd](https://arxiv.org/abs/2604.10493) (2026) | Builds an action-level reward dataset from SWE-Bench trajectories and a lightweight PRM for repository agents. | Direct learned coding-agent comparator, but its reported action guidance is not a matched proxy-vs-MC-value benchmark here. |
| [RUDDER](https://arxiv.org/abs/1806.07857) (2018/2019) | Studies delayed-reward return decomposition and reward redistribution. | Conceptual ancestor for temporal credit; does not supply this project's executable program-state signal. |

[RLEF](https://arxiv.org/abs/2410.02089) is adjacent work on learning to use
execution feedback for iterative code synthesis.
[ExecVerify](https://arxiv.org/abs/2603.11226) uses verifiable execution-trace
steps for *code execution reasoning*, not repository repair advantage.
These citations motivate comparisons; they do not establish a novelty gap or
license saying no prior paper addresses the same question. In particular,
the motivating note's proposed `public tests / executable q / learned PRM`
quality–cost Pareto comparison has **not** been run.

## Reading/claim rule

The current frozen-bank `q` is a behavioral probe, not a certified advantage,
learned PRM, or deployment-ready signal. The best-supported next *question* is
whether policy-conditioned semantic information improves **held-out** value
estimation beyond public tests; the current first-hit and P2 analysis does not
answer it decisively. Do not turn this reading map into a new PPO launch or a
positive project verdict without a separately authorized experiment.
