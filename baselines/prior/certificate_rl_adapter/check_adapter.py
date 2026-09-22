#!/usr/bin/env python3
"""Exact finite-tree checks for a certificate-compatible RL adapter.

This is an algebra/gradient audit, NOT an RL training experiment.
Dependency: numpy. Run: python check_adapter.py --trials 500 --seed 20260921
All policies are binary with one independent logit per decision state.
All expectations are computed by enumerating complete trajectories.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterator
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def paths(depth: int, p: np.ndarray) -> Iterator[tuple[float, list[int], list[int], int]]:
    def rec(node: int, level: int, prob: float, nodes: list[int], acts: list[int]):
        if level == depth:
            yield prob, nodes, acts, node
            return
        for a in (0, 1):
            pa = p[node] if a else 1.0 - p[node]
            yield from rec(2 * node + 1 + a, level + 1, prob * pa,
                           nodes + [node], acts + [a])
    yield from rec(0, 0, 1.0, [], [])


def value_function(depth: int, p: np.ndarray, leaf_reward: np.ndarray,
                   gamma: float) -> np.ndarray:
    n = 2**depth - 1
    v = np.zeros(2 ** (depth + 1) - 1)
    for s in reversed(range(n)):
        ch = (2 * s + 1, 2 * s + 2)
        qs = [float(leaf_reward[c - n]) if c >= n else gamma * v[c] for c in ch]
        v[s] = (1.0 - p[s]) * qs[0] + p[s] * qs[1]
    return v


def rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g = np.zeros_like(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        carry = rewards[t] + gamma * carry
        g[t] = carry
    return g


def gae(td: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    return rtg(td, gamma * lam)


def audit_random_trees(trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    maxima = {k: 0.0 for k in (
        'path_return_identity_error', 'monte_carlo_expected_gradient_error',
        'exact_td_identity_error', 'matched_approximate_gae_error',
        'finite_difference_gradient_error', 'residual_gradient_bound_excess')}
    trajectories = 0
    edges_on_trajectories = 0
    gamma_counts: dict[str, int] = {}
    for trial in range(trials):
        depth = int(rng.integers(2, 6))
        n = 2**depth - 1
        total = 2 ** (depth + 1) - 1
        gamma = float(rng.choice([1.0, 0.93, 0.7]))
        gamma_counts[str(gamma)] = gamma_counts.get(str(gamma), 0) + 1
        theta = rng.normal(0.0, 0.9, size=n)
        p = sigmoid(theta)
        rewards_at_leaves = rng.integers(0, 2, size=2**depth).astype(float)
        v = value_function(depth, p, rewards_at_leaves, gamma)
        phi = np.zeros(total)
        phi[:n] = -rng.uniform(0, 1, size=n)  # all terminal potentials = 0
        w = v - phi
        err = np.zeros(total)
        err[:n] = rng.uniform(-0.2, 0.2, size=n)
        vhat = v + err
        what = vhat - phi
        original_gradient = np.zeros(n)
        shaped_gradient = np.zeros(n)
        approximate_td_gradient = np.zeros(n)
        score_norm_budget = 0.0
        for prob, nodes, actions, leaf in paths(depth, p):
            trajectories += 1
            edges_on_trajectories += depth
            successors = nodes[1:] + [leaf]
            r = np.zeros(depth)
            r[-1] = rewards_at_leaves[leaf - n]
            rp = r + gamma * phi[successors] - phi[nodes]
            g, gp = rtg(r, gamma), rtg(rp, gamma)
            path_error = abs(gp[0] - (g[0] - phi[0]))
            maxima['path_return_identity_error'] = max(maxima['path_return_identity_error'], path_error)
            exact_td = r + gamma * v[successors] - v[nodes]
            exact_tdp = rp + gamma * w[successors] - w[nodes]
            maxima['exact_td_identity_error'] = max(maxima['exact_td_identity_error'],
                                                    float(np.max(np.abs(exact_td - exact_tdp))))
            td = r + gamma * vhat[successors] - vhat[nodes]
            tdp = rp + gamma * what[successors] - what[nodes]
            for lam in (0.0, 0.5, 0.95, 1.0):
                maxima['matched_approximate_gae_error'] = max(
                    maxima['matched_approximate_gae_error'],
                    float(np.max(np.abs(gae(td, gamma, lam) - gae(tdp, gamma, lam)))))
            for t, (s, a) in enumerate(zip(nodes, actions)):
                score = a - p[s]
                weight = prob * gamma**t
                original_gradient[s] += weight * score * g[t]
                shaped_gradient[s] += weight * score * gp[t]
                approximate_td_gradient[s] += weight * score * tdp[t]
                score_norm_budget += weight * abs(score)
        maxima['monte_carlo_expected_gradient_error'] = max(
            maxima['monte_carlo_expected_gradient_error'],
            float(np.max(np.abs(original_gradient - shaped_gradient))))
        epsilon = float(np.max(np.abs(err)))
        bound = gamma * epsilon * score_norm_budget
        bias = float(np.linalg.norm(approximate_td_gradient - original_gradient))
        maxima['residual_gradient_bound_excess'] = max(maxima['residual_gradient_bound_excess'],
                                                      max(0.0, bias - bound))
        # Independent derivative of the exact recursively computed objective.
        if trial < min(20, trials):
            fd = np.zeros(n)
            step = 1e-5
            for s in range(n):
                tp, tm = theta.copy(), theta.copy()
                tp[s] += step
                tm[s] -= step
                jp = value_function(depth, sigmoid(tp), rewards_at_leaves, gamma)[0]
                jm = value_function(depth, sigmoid(tm), rewards_at_leaves, gamma)[0]
                fd[s] = (jp - jm) / (2 * step)
            maxima['finite_difference_gradient_error'] = max(
                maxima['finite_difference_gradient_error'],
                float(np.max(np.abs(fd - original_gradient))))
    strict_keys = [k for k in maxima if k != 'finite_difference_gradient_error']
    assert all(maxima[k] < 1e-10 for k in strict_keys), maxima
    assert maxima['finite_difference_gradient_error'] < 1e-7, maxima
    return dict(trials=trials, seed=seed, depths=[2, 3, 4, 5],
                gamma_counts=gamma_counts, enumerated_trajectories=trajectories,
                transition_occurrences=edges_on_trajectories, maxima=maxima,
                assertions_passed=True)


def analytic_counterexamples() -> dict:
    p = 0.5
    true_gradient = p * (1 - p)
    # a0 is dummy; a1~Bernoulli(p); original terminal R=a1.
    # Move ALL reward to r0=a1 and set r1=0. Total unchanged,
    # but using only redistributed reward-to-go at t=1 loses all signal.
    leak = dict(true_objective=p, true_logit_gradient=true_gradient,
                noncausal_reward_to_go_gradient=0.0,
                total_reward_preserved_on_every_trajectory=True)

    # Exact action-dependent baseline c(a)=a. Subtraction alone kills gradient.
    naive_cv = sum((p if a else 1-p) * (a-p) * (a-a) for a in (0,1))
    cv = dict(true_gradient=true_gradient, naive_subtraction=naive_cv,
              required_expectation_correction=true_gradient,
              corrected_gradient=naive_cv + true_gradient)

    # Necessary temporary semantic regression vs a cosmetically good dead end.
    q0, qr, qc = -0.8, -1.0, -0.4
    v0, vr, vc = p, 1.0, 0.0
    w0, wr, wc = v0-q0, vr-qr, vc-qc
    dr, dc = qr-q0, qc-q0
    ar, ac = dr+wr-w0, dc+wc-w0
    grad_naive_direction = p*(1-p)*(-1) + (1-p)*(-p)*(1)
    grad_correct = p*(1-p)*ar + (1-p)*(-p)*ac
    detour = dict(semantic_effect_refactor=dr, semantic_effect_cosmetic=dc,
                   true_advantage_refactor=ar, true_advantage_cosmetic=ac,
                   exact_policy_gradient=grad_correct,
                   direction_only_policy_gradient=grad_naive_direction,
                   continuation_correction_refactor=wr-w0,
                   continuation_correction_cosmetic=wc-w0)
    assert abs(grad_correct-0.25) < 1e-12
    assert abs(grad_naive_direction+0.5) < 1e-12

    errors = [0.9, 0.6, 0.3, 0.9]
    differences = [errors[i]-errors[i+1] for i in range(3)]
    intervals = [(0.2,0.6),(0.2,0.4),(-0.7,-0.5)]
    assert all(l <= d <= u for d,(l,u) in zip(differences, intervals))
    cycle = dict(true_differences=differences, true_cycle_sum=sum(differences),
                 sign_reward_cycle_sum=sum(1 if d>0 else -1 for d in differences),
                 valid_certificate_intervals=intervals,
                 independently_chosen_midpoint_cycle_sum=sum((l+u)/2 for l,u in intervals))

    # Discrete curl zero for any single-valued fitted potential, even if inaccurate.
    fitted_potential = [-0.76, -0.55, -0.27]
    fitted_cycle = sum(fitted_potential[(i+1)%3]-fitted_potential[i] for i in range(3))
    assert abs(fitted_cycle) < 1e-12
    cycle['single_valued_potential_cycle_sum'] = fitted_cycle
    return dict(noncausal_redistribution=leak, action_dependent_baseline=cv,
                necessary_detour=detour, reward_cycling=cycle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('results.json'))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    report = {'scope': 'Exact algebra and expected-gradient checks; no trained RL agent.',
              'random_tree_audit': audit_random_trees(args.trials,args.seed),
              'counterexamples': analytic_counterexamples()}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
