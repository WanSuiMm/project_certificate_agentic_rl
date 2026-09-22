#!/usr/bin/env python3
"""Microbenchmark: common-prefix/common-suffix exact replay is a strong baseline.

These timings are local implementation observations, not scalable-system claims.
Setup costs are INCLUDED. Each method sees all the same fixtures and edits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from experiment import backward, execute, XS, TARGET, N, LOW


def exact_cached(fixtures: list[dict]) -> tuple[int, int]:
    calls, checksum = 0, 0
    for f in fixtures:
        p = np.asarray(f['tables'], dtype=np.int64)
        prefix = execute(p)
        suffix = np.empty((len(p) + 1, N), dtype=np.int64)
        suffix[-1] = XS
        for j in range(len(p) - 1, -1, -1):
            suffix[j] = suffix[j + 1, p[j]]
        for edit in f['edits']:
            j = edit['stage']; new = np.asarray(edit['new_table'], dtype=np.int64)
            old_y = p[j, prefix[j]]; new_y = new[prefix[j]]
            old_e = suffix[j + 1, old_y] // LOW != TARGET
            new_e = suffix[j + 1, new_y] // LOW != TARGET
            delta = int(old_e.sum()) - int(new_e.sum())
            calls += delta != 0
            checksum += delta
    return calls, checksum


def paired_abstract(fixtures: list[dict], block: int = 4) -> tuple[int, int]:
    calls, checksum = 0, 0
    for f in fixtures:
        p = np.asarray(f['tables'], dtype=np.int64)
        prefix = execute(p)
        lo, hi = backward(p, block)
        for edit in f['edits']:
            j = edit['stage']; new = np.asarray(edit['new_table'], dtype=np.int64)
            old_y = p[j, prefix[j]]; new_y = new[prefix[j]]
            changed = old_y != new_y
            lower = int((lo[j + 1, TARGET, old_y // block] -
                         hi[j + 1, TARGET, new_y // block])[changed].sum())
            upper = int((hi[j + 1, TARGET, old_y // block] -
                         lo[j + 1, TARGET, new_y // block])[changed].sum())
            calls += lower > 0 or upper < 0
            checksum += lower + upper
    return calls, checksum


def main(root: Path, repeats: int) -> None:
    fixtures = json.loads((root / 'fixtures.json').read_text())
    # Warm both implementations; then alternate order to reduce timing bias.
    exact_cached(fixtures); paired_abstract(fixtures)
    times = {'exact_cached': [], 'paired_abstract_block4': []}
    outputs = {}
    for r in range(repeats):
        methods = [('exact_cached', exact_cached), ('paired_abstract_block4', paired_abstract)]
        if r % 2:
            methods.reverse()
        for name, fn in methods:
            t = perf_counter(); outputs[name] = fn(fixtures)
            times[name].append(perf_counter() - t)
    result = {'repeats': repeats, 'setup_cost_included': True,
              'timings_seconds': times,
              'median_seconds': {k: float(np.median(v)) for k, v in times.items()},
              'direction_calls_and_checksums': outputs,
              'warning': 'Tiny explicit transition tables favor exact summaries. '
                         'This benchmark establishes no certificate-speed advantage.'}
    (root / 'cost_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, nargs='?', default=Path('results'))
    p.add_argument('--repeats', type=int, default=7)
    a=p.parse_args(); main(a.results, a.repeats)
