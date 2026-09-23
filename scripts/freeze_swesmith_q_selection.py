#!/usr/bin/env python3
"""Freeze a q-qualified, policy-outcome-blind SWE-smith task split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SEED = "swesmith-q-first-split-v01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--q-results", required=True, type=Path)
    parser.add_argument("--gold-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="four-task engineering smoke only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = {row["instance_id"]: row for line in args.tasks.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line))}
    q = {row["instance_id"]: row for line in args.q_results.read_text(encoding="utf-8").splitlines()
         if (row := json.loads(line)) and row["status"] == "q_valid"
         and row["instance_id"] in tasks}
    gold = {row["instance_id"] for line in args.gold_results.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)) and row["status"] == "gold_valid"}
    q = {task_id: row for task_id, row in q.items() if task_id in gold}
    count = 4 if args.smoke else next((n for n in (64, 52, 40) if len(q) >= n), 0)
    if count == 0 or len(q) < count:
        raise RuntimeError(f"insufficient q-valid tasks: {len(q)} (need 40 formal or 4 smoke)")
    rank = lambda task_id: hashlib.sha256(f"{SEED}\0{task_id}".encode()).hexdigest()
    ids = sorted(q, key=rank)[:count]
    train_count = {64: 48, 52: 40, 40: 32, 4: 3}[count]
    splits = {task_id: "train" if i < train_count else "heldout" for i, task_id in enumerate(ids)}
    manifest = {"status": "q_first_frozen", "scope": "engineering_smoke" if args.smoke else "formal_survival",
                "count": count, "seed": SEED, "ids": ids, "splits": splits,
                "train_count": train_count, "heldout_count": count - train_count,
                "images": dict(Counter(tasks[task_id]["image_name"] for task_id in ids)),
                "q_initial": {task_id: q[task_id]["q_initial"] for task_id in ids},
                "bank_sha256": {task_id: q[task_id]["bank_sha256"] for task_id in ids},
                "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                "q_results_sha256": hashlib.sha256(args.q_results.read_bytes()).hexdigest(),
                "gold_results_sha256": hashlib.sha256(args.gold_results.read_bytes()).hexdigest(),
                "selection_does_not_use": ["policy_rollout", "terminal_solve_outcome", "F2P/P2P_inputs"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"ids", "splits", "bank_sha256", "q_initial"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
