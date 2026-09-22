#!/usr/bin/env python3
"""Independent stdlib-only replay of stored fixtures and all reported bounds.
Does not import the analyzer or numpy. Fail-fast on any discrepancy.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def outputs(tables: list[list[int]]) -> list[int]:
    ans = []
    for x in range(64):
        y = x
        for table in tables:
            y = table[y]
        ans.append(y)
    return ans


def main(root: Path) -> None:
    fixtures = json.loads((root / 'fixtures.json').read_text(encoding='utf-8'))
    truths = {}
    for program in fixtures:
        tables = program['tables']
        old_out = outputs(tables)
        old_loss = [int(y // 16 != x // 16) for x, y in enumerate(old_out)]
        for eid, edit in enumerate(program['edits']):
            new_tables = [list(t) for t in tables]
            new_tables[edit['stage']] = edit['new_table']
            new_out = outputs(new_tables)
            new_loss = [int(y // 16 != x // 16) for x, y in enumerate(new_out)]
            truths[(program['pid'], eid)] = (sum(old_loss), sum(new_loss))
    nrows = 0
    with (root / 'edits.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            before, after = truths[(int(row['pid']), int(row['edit']))]
            assert before == int(row['before_E_num'])
            assert after == int(row['after_E_num'])
            assert int(row['before_L_num']) <= before <= int(row['before_U_num'])
            assert int(row['after_L_num']) <= after <= int(row['after_U_num'])
            delta = before - after
            for prefix in ('global', 'marginal', 'paired'):
                lower = int(row[prefix + '_delta_L_num'])
                upper = int(row[prefix + '_delta_U_num'])
                assert lower <= delta <= upper
                asserted = int(row[prefix + '_sign'])
                true_sign = (delta > 0) - (delta < 0)
                assert asserted == 0 or asserted == true_sign
            nrows += 1
    report = {'independent_oracle': 'stdlib-only scalar interpreter',
              'unique_program_edits_replayed': len(truths),
              'analyzer_configurations_checked': nrows,
              'violations': 0}
    (root / 'independent_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, nargs='?', default=Path('results'))
    main(p.parse_args().results)
