"""Summarize the structural properties of a frozen SWE-smith selection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "resolved": [row for row in rows if row["resolved"]],
        "unresolved": [row for row in rows if not row["resolved"]],
    }
    output: dict[str, Any] = {}
    for name, group in groups.items():
        action_counts: list[float] = []
        edit_counts: list[float] = []
        edit_fractions: list[float] = []
        for row in group:
            messages = json.loads(row["messages"])
            actions = sum(message.get("message_type") == "action" for message in messages)
            edits = int(row["structured_edit_count"])
            action_counts.append(actions)
            edit_counts.append(edits)
            edit_fractions.append(edits / actions if actions else 0.0)
        output[name] = {
            "action_count": describe(action_counts),
            "structured_edit_count": describe(edit_counts),
            "structured_edit_fraction": describe(edit_fractions),
        }
    output["repository_count"] = len({row["repository"] for row in rows})
    output["model_counts"] = dict(sorted(Counter(row["model"] for row in rows).items()))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    summary = summarize(rows)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
