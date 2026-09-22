#!/usr/bin/env python3
"""Finite-program certificate audit (not a coding-LLM/RL benchmark).

All soundness assertions use integer numerators, not float tolerances.
Uniform input distribution on {0,...,63}; four output classes.
The task is to preserve the upper two bits; the low four bits are unconstrained.
The analyzer uses sound block partitions and min/max backward propagation.
It never receives the uncorrupted implementation or exact final losses.
Exact enumeration is a SEPARATE audit oracle.

Run: python experiment.py --out results
Dependency: numpy. Runtime: typically seconds to a few minutes on a CPU.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np

N = 64
C = 4
LOW = N // C
XS = np.arange(N, dtype=np.int64)
TARGET = XS // LOW
BLOCKS = (16, 8, 4, 2, 1)
DEPTHS = (4, 8, 16)


def dump_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def backward(program: np.ndarray, block: int, start: int = 0,
             reuse: tuple[np.ndarray, np.ndarray] | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """Sound backward min/max enclosures for each label and abstract cell.

    If reuse is given, stages > start are unchanged and are reused; only the
    changed stage and its preceding stages are recomputed.
    """
    depth = len(program)
    cells = N // block
    if reuse is None:
        lo = np.zeros((depth + 1, C, cells), dtype=np.int64)
        hi = np.zeros_like(lo)
        terminal = (XS[None, :] // LOW != np.arange(C)[:, None]).astype(np.int64)
        lo[depth] = terminal.reshape(C, cells, block).min(axis=2)
        hi[depth] = terminal.reshape(C, cells, block).max(axis=2)
        top = depth - 1
    else:
        lo, hi = (a.copy() for a in reuse)
        top = start
    for j in range(top, -1, -1):
        successors = program[j] // block
        lo[j] = lo[j + 1][:, successors].reshape(C, cells, block).min(axis=2)
        hi[j] = hi[j + 1][:, successors].reshape(C, cells, block).max(axis=2)
    return lo, hi


def execute(program: np.ndarray) -> np.ndarray:
    states = [XS.copy()]
    for table in program:
        states.append(table[states[-1]])
    return np.asarray(states)


def generate_program(rng: np.random.Generator, depth: int,
                     family: str) -> np.ndarray:
    """Base operations are total finite-word functions with guarded mutations.

    local: preserve most low-bit locality.
    mixing: mix low bits heavily. Both preserve task classes before mutation.
    """
    tables = []
    low = XS % LOW
    high = XS - low
    for _ in range(depth):
        if family == 'local':
            bit = int(rng.choice([1, 2, 4]))
            base = high + (low ^ bit)
        elif family == 'mixing':
            a = int(rng.choice([3, 5, 7, 11, 13, 15]))
            b = int(rng.integers(0, LOW))
            base = high + ((a * low + b) % LOW)
        else:
            raise ValueError(f'Unknown family: {family}')
        tables.append(base)
    program = np.asarray(tables, dtype=np.int64)
    # Corrupt one to four stages with guarded assignments, without giving the
    # original program to the certificate analyzer or the edit proposal routine.
    for j in rng.choice(depth, size=min(depth, int(rng.integers(1, 5))), replace=False):
        chosen = rng.choice(N, size=int(rng.choice([1, 2, 4, 8])), replace=False)
        delta = int(rng.integers(1, C)) * LOW
        program[j, chosen] = (program[j, chosen] + delta) % N
    return program


def candidate_edit(rng: np.random.Generator, program: np.ndarray,
                   ordinal: int) -> tuple[int, np.ndarray]:
    j = int(rng.integers(len(program)))
    new = program[j].copy()
    changed = rng.choice(N, size=int(rng.choice([1, 2, 4, 8])), replace=False)
    if ordinal % 3 == 0:
        # This proposal tries the known task-class invariant locally. It is NOT
        # guaranteed beneficial globally (other defects may compensate for it).
        new[changed] = (changed // LOW) * LOW + new[changed] % LOW
    else:
        new[changed] = (new[changed] + int(rng.integers(1, C)) * LOW) % N
    return j, new


def sign_bounds(lower: int, upper: int) -> int:
    if lower > 0:
        return 1
    if upper < 0:
        return -1
    return 0


def audit_residual(program: np.ndarray, states: np.ndarray, block: int,
                   bounds: tuple[np.ndarray, np.ndarray]) -> dict:
    """q is scaled by 2: q = lower+upper, terminal q=2*loss.

    Tests the signed telescoping identity and an occupancy-weighted residual
    enclosure. This is an algebraic identity, NOT a free way of avoiding
    program execution; the audit deliberately pays for full execution.
    """
    lo, hi = bounds
    depth = len(program)
    q = (lo + hi)[:, :, XS // block].copy()
    q[depth] = 2 * (XS[None, :] // LOW != np.arange(C)[:, None])
    start = q[0, TARGET, XS]
    total = start.copy()
    lower = int(start.sum())
    upper = lower
    rho_l1 = 0
    for j in range(depth):
        residual = q[j + 1][:, program[j]] - q[j]
        local = residual[TARGET, states[j]]
        total += local
        rho_l1 += int(np.abs(local).sum())
        rlo = residual.reshape(C, N // block, block).min(axis=2)
        rhi = residual.reshape(C, N // block, block).max(axis=2)
        lower += int(rlo[TARGET, states[j] // block].sum())
        upper += int(rhi[TARGET, states[j] // block].sum())
    terminal = 2 * (states[-1] // LOW != TARGET)
    assert np.array_equal(total, terminal), 'Signed identity violated'
    true_num = int(terminal.sum())
    assert lower <= true_num <= upper, 'Residual enclosure violated'
    assert abs(true_num - int(start.sum())) <= rho_l1, 'L1 bound violated'
    return {'residual_interval_width': (upper - lower) / (2 * N),
            'signed_sum': (true_num - int(start.sum())) / (2 * N),
            'absolute_residual_sum': rho_l1 / (2 * N)}


def deterministic_counterexamples() -> dict:
    # (1) A non-task-sufficient abstraction alpha maps everything to a single
    # point. Abstract analyzer error and abstract mismatch are both zero.
    alpha_false_bound = {'abstract_mismatch': 0, 'analyzer_imprecision': 0,
                         'true_task_error': 1}
    assert alpha_false_bound['true_task_error'] > 0
    # (2) Compensating bugs: +1/4 then -1/4, reference modules are identity.
    # Removing the first defect decreases the norm-sum U yet worsens E.
    cancellation = {'before_E': 0, 'before_U': 0.5,
                    'after_E': 0.25, 'after_U': 0.25}
    assert cancellation['after_U'] < cancellation['before_U']
    assert cancellation['after_E'] > cancellation['before_E']
    # (3) Analyzer refinement can lower U while P and true E are fixed.
    refinement = {'E': 0.25, 'coarse_U': 1.0, 'fine_U': 0.25,
                  'naive_reward_without_code_change': 0.75}
    # (4) A two-edit repair requires temporarily larger error.
    # output 0 for (1,1), 1 for (0,0), 2 otherwise; target output is 0.
    # Only one flag may be toggled in an edit.
    errors = {(0, 0): 0.5, (1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.0}
    distances = {(1, 1): 0, (1, 0): 1, (0, 1): 1, (0, 0): 2}
    assert all(errors[s] > errors[(0, 0)] for s in [(1, 0), (0, 1)])
    assert distances[(1, 0)] < distances[(0, 0)]
    barrier = {'00': {'error': 0.5, 'optimal_remaining_edits': 2},
               '10': {'error': 1.0, 'optimal_remaining_edits': 1},
               '01': {'error': 1.0, 'optimal_remaining_edits': 1},
               '11': {'error': 0.0, 'optimal_remaining_edits': 0}}
    # (5) Check shaping identity exactly with rational arithmetic and terminal
    # potential zero. Include some deliberately nonmonotone error trajectories.
    from fractions import Fraction as F
    paths = [[F(1, 2), F(1), F(0)], [F(1, 2), F(1, 4), F(1, 2), F(0)],
             [F(1, 2), F(1, 2), F(3, 4), F(1)]]
    for gamma in (F(1), F(9, 10)):
        for vals in paths:
            phi = [-v for v in vals[:-1]] + [F(0)]
            shaped = sum(gamma**t * (gamma * phi[t + 1] - phi[t])
                         for t in range(len(phi) - 1))
            assert shaped == -phi[0]
    # (6) Residual localization is not unique: the same terminal error can be
    # encoded entirely in q_0 or entirely in the last residual.
    q_a = [1, 1, 1, 1]
    q_b = [0, 0, 0, 1]
    r_a = [v - u for u, v in zip(q_a, q_a[1:])]
    r_b = [v - u for u, v in zip(q_b, q_b[1:])]
    assert q_a[0] + sum(r_a) == q_b[0] + sum(r_b) == 1
    assert r_a != r_b
    gauge = {'same_true_error': 1, 'q_A': q_a, 'residual_A': r_a,
             'q_B': q_b, 'residual_B': r_b,
             'lesson': 'Residual location is not an invariant causal attribution.'}
    return {'residual_localization_depends_on_auxiliary_q': gauge,
            'abstraction_nonidentifiability': alpha_false_bound,
            'upper_bound_decrease_is_not_improvement': cancellation,
            'analysis_refinement_is_not_program_repair': refinement,
            'true_error_is_not_remaining_repair_distance': barrier,
            'potential_shaping_exact_rational_checks': 6}


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for family in ('local', 'mixing', 'all'):
        for block in BLOCKS:
            rr = [r for r in rows if r['block'] == block and
                  (family == 'all' or r['family'] == family)]
            effect = np.asarray([r['delta_num'] for r in rr])
            nonzero = effect != 0
            def method(prefix: str) -> dict:
                calls = np.asarray([r[prefix + '_sign'] for r in rr])
                asserted = calls != 0
                false = asserted & (calls != np.sign(effect))
                return {'directional_calls': int(asserted.sum()),
                        'false_directional_calls': int(false.sum()),
                        'all_edit_coverage': float(asserted.mean()),
                        'nonzero_effect_coverage': float((asserted & nonzero).sum() /
                                                       max(1, nonzero.sum())),
                        'positive_effect_calls': int(((calls == 1) & (effect > 0)).sum()),
                        'positive_effect_total': int((effect > 0).sum()),
                        'negative_effect_calls': int(((calls == -1) & (effect < 0)).sum()),
                        'negative_effect_total': int((effect < 0).sum())}
            positive_upper = np.asarray([r['upper_drop_num'] > 0 for r in rr])
            worsened = effect < 0
            summaries.append({'family': family, 'block': block, 'n_edits': len(rr),
                              'n_nonzero_effect': int(nonzero.sum()),
                              'n_positive_effect': int((effect > 0).sum()),
                              'n_negative_effect': int((effect < 0).sum()),
                              'global_interval': method('global'),
                              'prefix_marginal_interval': method('marginal'),
                              'paired_interval': method('paired'),
                              'eight_tests': method('tests'),
                              'midpoint': method('midpoint'),
                              'mean_global_delta_width': float(np.mean([r['global_width_num'] / N for r in rr])),
                              'mean_marginal_delta_width': float(np.mean([r['marginal_width_num'] / N for r in rr])),
                              'mean_paired_delta_width': float(np.mean([r['paired_width_num'] / N for r in rr])),
                              'mean_pair_affected_input_fraction': float(np.mean([r['affected_n'] / N for r in rr])),
                              'upper_drop_positive_calls': int(positive_upper.sum()),
                              'upper_drop_calls_that_worsen_true_error': int((positive_upper & worsened).sum()),
                              'mean_global_U': float(np.mean([r['before_U_num'] / N for r in rr]))})
    return summaries


def run(out: Path, programs_per_cell: int, edits_per_program: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rng = np.random.default_rng(seed)
    rows, programs, residual_rows = [], [], []
    worst_upper_example = None
    checks = {'global_intervals': 0, 'delta_intervals': 0, 'pathwise_residual_identities': 0,
              'residual_global_intervals': 0, 'paired_width_dominance': 0,
              'interval_refinement_monotonicity': 0, 'violations': 0}
    for family in ('local', 'mixing'):
        for depth in DEPTHS:
            for rep in range(programs_per_cell):
                pid = len(programs)
                p = generate_program(rng, depth, family)
                states = execute(p)
                before_losses = (states[-1] // LOW != TARGET).astype(np.int64)
                before_num = int(before_losses.sum())
                bounds = {block: backward(p, block) for block in BLOCKS}
                tests = np.sort(rng.choice(N, size=8, replace=False))
                edits = []
                for block in BLOCKS:
                    residual_rows.append({'pid': pid, 'family': family, 'depth': depth,
                                          'block': block, **audit_residual(p, states, block, bounds[block])})
                    checks['pathwise_residual_identities'] += N
                    checks['residual_global_intervals'] += 1
                for edit_no in range(edits_per_program):
                    j, table = candidate_edit(rng, p, edit_no)
                    pnew = p.copy()
                    pnew[j] = table
                    after_states = execute(pnew)
                    after_losses = (after_states[-1] // LOW != TARGET).astype(np.int64)
                    after_num = int(after_losses.sum())
                    delta_num = before_num - after_num
                    old_y, new_y = p[j, states[j]], table[states[j]]
                    affected = old_y != new_y
                    prev_pair = None
                    for block in BLOCKS:
                        lo, hi = bounds[block]
                        ln, hn = backward(pnew, block, start=j, reuse=(lo, hi))
                        bL = int(lo[0, TARGET, XS // block].sum())
                        bU = int(hi[0, TARGET, XS // block].sum())
                        aL = int(ln[0, TARGET, XS // block].sum())
                        aU = int(hn[0, TARGET, XS // block].sum())
                        assert bL <= before_num <= bU
                        assert aL <= after_num <= aU
                        checks['global_intervals'] += 2
                        glo, ghi = bL - aU, bU - aL
                        lo_old = lo[j + 1, TARGET, old_y // block]
                        hi_old = hi[j + 1, TARGET, old_y // block]
                        lo_new = lo[j + 1, TARGET, new_y // block]
                        hi_new = hi[j + 1, TARGET, new_y // block]
                        ml, mh = int((lo_old - hi_new).sum()), int((hi_old - lo_new).sum())
                        pl = int((lo_old - hi_new)[affected].sum())
                        ph = int((hi_old - lo_new)[affected].sum())
                        assert glo <= delta_num <= ghi
                        assert ml <= delta_num <= mh
                        assert pl <= delta_num <= ph
                        checks['delta_intervals'] += 3
                        assert ph - pl <= mh - ml
                        assert ml <= pl and ph <= mh
                        checks['paired_width_dominance'] += 1
                        if prev_pair is not None:
                            assert pl >= prev_pair[0] and ph <= prev_pair[1]
                            checks['interval_refinement_monotonicity'] += 1
                        prev_pair = (pl, ph)
                        test_delta = int((before_losses[tests] - after_losses[tests]).sum())
                        mid_delta2 = bL + bU - aL - aU
                        row = {'pid': pid, 'family': family, 'depth': depth, 'edit': edit_no,
                               'stage': j, 'block': block, 'before_E_num': before_num,
                               'after_E_num': after_num, 'delta_num': delta_num,
                               'before_L_num': bL, 'before_U_num': bU,
                               'after_L_num': aL, 'after_U_num': aU,
                               'upper_drop_num': bU - aU,
                               'global_delta_L_num': glo, 'global_delta_U_num': ghi,
                               'global_width_num': ghi - glo,
                               'global_sign': sign_bounds(glo, ghi),
                               'marginal_delta_L_num': ml, 'marginal_delta_U_num': mh,
                               'marginal_width_num': mh - ml,
                               'marginal_sign': sign_bounds(ml, mh),
                               'paired_delta_L_num': pl, 'paired_delta_U_num': ph,
                               'paired_width_num': ph - pl,
                               'paired_sign': sign_bounds(pl, ph),
                               'tests_sign': int(np.sign(test_delta)),
                               'tests_delta_num_out_of_8': test_delta,
                               'midpoint_sign': int(np.sign(mid_delta2)),
                               'affected_n': int(affected.sum())}
                        rows.append(row)
                        if bU > aU and delta_num < 0 and worst_upper_example is None:
                            worst_upper_example = dict(row)
                        if block == 1:
                            assert bL == bU == before_num
                            assert aL == aU == after_num
                            assert pl == ph == delta_num
                    edits.append({'stage': j, 'new_table': table.tolist()})
                programs.append({'pid': pid, 'family': family, 'depth': depth,
                                 'tables': p.tolist(), 'tests': tests.tolist(), 'edits': edits})
    summaries = summarize(rows)
    with (out / 'edits.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (out / 'residuals.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(residual_rows[0]))
        w.writeheader(); w.writerows(residual_rows)
    # Full generated fixtures are provided to make the tested universe explicit.
    dump_json(out / 'fixtures.json', programs)
    dump_json(out / 'counterexamples.json', {
        **deterministic_counterexamples(),
        'fixed_analyzer_upper_drop_worsening_witness': worst_upper_example})
    metadata = {'seed': seed, 'states': N, 'classes': C, 'depths': DEPTHS,
                'families': ['local', 'mixing'], 'blocks': BLOCKS,
                'programs': len(programs), 'edits_per_program': edits_per_program,
                'unique_edits': len(programs) * edits_per_program,
                'paired_analyzer_configurations': len(rows),
                'checks': checks, 'runtime_seconds': perf_counter() - started,
                'python': platform.python_version(), 'numpy': np.__version__,
                'oracle': 'Exact enumeration of all 64 inputs, only for audit',
                'guarantee_scope': 'Finite deterministic programs; fixed task and uniform input measure',
                'cost_note': 'No speedup claimed. Exact prefixes and abstract transition tables are paid for. '
                             'In this tiny world exact enumeration is an extremely strong baseline.',
                'learning_note': 'No neural-network or reinforcement-learning model was trained.'}
    dump_json(out / 'summary.json', {'metadata': metadata, 'summaries': summaries})
    print(json.dumps(metadata, indent=2))
    print('\nAll families: nonzero-effect direction coverage / false calls')
    for r in summaries:
        if r['family'] == 'all':
            print('block', r['block'], 'nonzero', r['n_nonzero_effect'],
                  'global', r['global_interval'], 'paired', r['paired_interval'],
                  'tests', r['eight_tests'], 'false upper-drop',
                  r['upper_drop_calls_that_worsen_true_error'], '/', r['upper_drop_positive_calls'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--programs-per-cell', type=int, default=40,
                        help='Programs for each family/depth combination; default yields 240 programs')
    parser.add_argument('--edits-per-program', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    if args.programs_per_cell < 1 or args.edits_per_program < 1:
        parser.error('Sample counts must be positive')
    run(args.out, args.programs_per_cell, args.edits_per_program, args.seed)

if __name__ == '__main__':
    main()
