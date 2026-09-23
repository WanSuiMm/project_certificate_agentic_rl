#!/usr/bin/env python3
"""Make a path-free, reviewable aggregate from exact-bound replay summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def aggregate_summaries(summaries: list[dict], expected_tasks: int = 25) -> dict:
    labels: Counter[str] = Counter()
    rows = []
    for summary in sorted(summaries, key=lambda item: item["traj_id"].casefold()):
        curve = [
            {key: state[key] for key in ("state_index", "F", "R", "valid")}
            for state in summary["curve"]
        ]
        local_labels = Counter(item["label"] for item in summary["transitions"])
        labels.update(local_labels)
        initialization = summary["initialization"]
        endpoint_gate = bool(
            initialization["git_apply_check_returncode"] == 0
            and initialization["git_apply_returncode"] == 0
            and curve
            and curve[0]["valid"]
            and curve[0]["F"] > 0
            and curve[-1]["valid"]
            and summary["endpoint_agrees"]
        )
        rows.append(
            {
                "traj_id": summary["traj_id"],
                "instance_id": summary["instance_id"],
                "declared_resolved": summary["declared_resolved"],
                "observed_resolved": summary["observed_resolved"],
                "endpoint_agrees": summary["endpoint_agrees"],
                "endpoint_gate": endpoint_gate,
                "initialization_check_returncode": initialization[
                    "git_apply_check_returncode"
                ],
                "initialization_apply_returncode": initialization["git_apply_returncode"],
                "state_count": summary["state_count"],
                "valid_state_count": sum(state["valid"] for state in curve),
                "mutation_count": summary["mutation_count"],
                "transition_counts": dict(sorted(local_labels.items())),
                "curve": curve,
            }
        )
    complete = len(rows) == expected_tasks and len({r["traj_id"] for r in rows}) == len(rows)
    return {
        "protocol": "exact25_intermediate_v01",
        "expected_task_count": expected_tasks,
        "observed_task_count": len(rows),
        "complete": complete,
        "endpoint_agreement_count": sum(r["endpoint_agrees"] for r in rows),
        "endpoint_gate_pass_count": sum(r["endpoint_gate"] for r in rows),
        "all_endpoint_gates_pass": bool(complete and all(r["endpoint_gate"] for r in rows)),
        "state_count": sum(r["state_count"] for r in rows),
        "valid_state_count": sum(r["valid_state_count"] for r in rows),
        "transition_counts": dict(sorted(labels.items())),
        "trajectories_with_negative_transition": sum(
            r["transition_counts"].get("negative", 0) > 0 for r in rows
        ),
        "trajectories_with_invalid_state": sum(
            r["valid_state_count"] < r["state_count"] for r in rows
        ),
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=25)
    args = parser.parse_args()
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("*/summary.json"))
    ]
    aggregate = aggregate_summaries(summaries, args.expected_tasks)
    output = args.run_dir / "aggregate_summary.json"
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"{output}: {aggregate['endpoint_gate_pass_count']}/{aggregate['observed_task_count']} endpoint gates")


if __name__ == "__main__":
    main()
