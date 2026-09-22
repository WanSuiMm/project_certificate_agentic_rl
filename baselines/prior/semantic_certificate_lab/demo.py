#!/usr/bin/env python3
"""Reproduce the fixed-analyzer counterexample reported in the study."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from experiment import backward, execute, XS, TARGET, LOW, N

root = Path(__file__).resolve().parent / 'results'
fixtures = json.loads((root / 'fixtures.json').read_text())
witness = json.loads((root / 'counterexamples.json').read_text())[
    'fixed_analyzer_upper_drop_worsening_witness']
f = fixtures[witness['pid']]
e = f['edits'][witness['edit']]
p = np.asarray(f['tables'], dtype=np.int64)
pnew = p.copy(); pnew[e['stage']] = e['new_table']
block = witness['block']; j = e['stage']
s = execute(p); sn = execute(pnew)
lo, hi = backward(p, block); ln, hn = backward(pnew, block)
y = p[j, s[j]]; yn = pnew[j, s[j]]; changed = y != yn
lower = int((lo[j+1,TARGET,y//block] - hi[j+1,TARGET,yn//block])[changed].sum())
upper = int((hi[j+1,TARGET,y//block] - lo[j+1,TARGET,yn//block])[changed].sum())
before = int((s[-1]//LOW != TARGET).sum()); after = int((sn[-1]//LOW != TARGET).sum())
bU = int(hi[0,TARGET,XS//block].sum()); aU = int(hn[0,TARGET,XS//block].sum())
assert lower <= before-after <= upper
print(f'Actual errors: {before}/{N} -> {after}/{N}')
print(f'Sound upper bounds: {bU}/{N} -> {aU}/{N}')
print(f'Naive upper-bound decrease: {(bU-aU)/N:+.6f}')
print(f'Actual improvement: {(before-after)/N:+.6f}')
print(f'Paired improvement certificate: [{lower}/{N}, {upper}/{N}]')
