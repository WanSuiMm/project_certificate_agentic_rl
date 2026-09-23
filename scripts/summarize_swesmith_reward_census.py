#!/usr/bin/env python3
"""Summarize frozen one-edit reward information; never infer policy improvement."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path


def summarize(rows: list[dict], task_ids: list[str]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)
    complete = []
    for task_id in task_ids:
        block = grouped[task_id]
        if len(block) == 16 and {row["sample"] for row in block} == set(range(16)):
            complete.append((task_id, block))
    result = {"complete_tasks": len(complete), "expected_tasks": len(task_ids),
              "candidates": 16 * len(complete), "executable": 0, "solved": 0,
              "tasks_with_terminal_variance": 0, "tasks_with_test_reward_variance": 0,
              "tasks_with_semantic_reward_variance": 0,
              "tasks_with_semantic_variance_but_test_tied": 0,
              "unresolved_executable": 0, "unresolved_test_tie_pairs": 0,
              "unresolved_ties_split_by_semantic": 0,
              "tasks_with_unresolved_ties_split": 0,
              "task_details": []}
    for task_id, block in complete:
        scored = [row for row in block if row["status"] == "scored"]
        unresolved = [row for row in scored if not row["score"]["solved"]]
        test_values = {row["test_reward"] for row in scored}
        semantic_values = {row["semantic_reward"] for row in scored}
        terminal_values = {row["score"]["solved"] for row in scored}
        pairs = [(a, b) for a, b in combinations(unresolved, 2)
                 if a["test_reward"] == b["test_reward"]]
        split = sum(a["semantic_reward"] != b["semantic_reward"] for a, b in pairs)
        solved = sum(row["score"]["solved"] for row in scored)
        result["executable"] += len(scored)
        result["solved"] += solved
        result["unresolved_executable"] += len(unresolved)
        result["tasks_with_terminal_variance"] += len(terminal_values) > 1
        result["tasks_with_test_reward_variance"] += len(test_values) > 1
        result["tasks_with_semantic_reward_variance"] += len(semantic_values) > 1
        result["tasks_with_semantic_variance_but_test_tied"] += (
            len(test_values) == 1 and len(semantic_values) > 1)
        result["unresolved_test_tie_pairs"] += len(pairs)
        result["unresolved_ties_split_by_semantic"] += split
        result["tasks_with_unresolved_ties_split"] += split > 0
        result["task_details"].append({"instance_id": task_id, "executable": len(scored),
                                       "solved": solved, "test_reward_values": len(test_values),
                                       "semantic_reward_values": len(semantic_values),
                                       "unresolved_test_ties": len(pairs),
                                       "ties_split_by_semantic": split})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()
    results_bytes = args.results.read_bytes()
    rows = [json.loads(line) for line in results_bytes.splitlines() if line]
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    summary = summarize(rows, selection["ids"])
    summary["results_sha256"] = hashlib.sha256(results_bytes).hexdigest()
    summary["selection_sha256"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    summary["recorded_rows"] = len(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
