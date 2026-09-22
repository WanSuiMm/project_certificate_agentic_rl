#!/usr/bin/env python3
"""Aggregate compact Moto intermediate replay summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=5)
    args = parser.parse_args()

    paths = sorted(args.run_dir.glob("getmoto__*/summary.json"))
    summaries: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    transition_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        local_counts = Counter(item["label"] for item in summary["transitions"])
        transition_counts.update(local_counts)
        curve = summary["curve"]
        endpoint_gate = bool(
            summary["initialization"]["git_apply_check_returncode"] == 0
            and summary["initialization"]["git_apply_returncode"] == 0
            and curve
            and curve[0]["valid"]
            and curve[0]["F"] > 0
            and curve[-1]["valid"]
            and summary["endpoint_agrees"]
        )
        rows.append(
            {
                "traj_id": summary["traj_id"],
                "declared_resolved": summary["declared_resolved"],
                "observed_resolved": summary["observed_resolved"],
                "endpoint_agrees": summary["endpoint_agrees"],
                "endpoint_gate": endpoint_gate,
                "state_count": summary["state_count"],
                "valid_state_count": sum(state["valid"] for state in curve),
                "mutation_count": summary["mutation_count"],
                "attempted_mutation_count": summary["attempted_mutation_count"],
                "initial": {key: curve[0][key] for key in ("F", "R", "valid")},
                "final": {key: curve[-1][key] for key in ("F", "R", "valid")},
                "transition_counts": dict(sorted(local_counts.items())),
                "curve": curve,
            }
        )

    complete = len(summaries) == args.expected_tasks
    aggregate = {
        "protocol": "moto_intermediate_v01",
        "expected_task_count": args.expected_tasks,
        "observed_task_count": len(summaries),
        "complete": complete,
        "endpoint_agreement_count": sum(row["endpoint_agrees"] for row in rows),
        "endpoint_gate_pass_count": sum(row["endpoint_gate"] for row in rows),
        "PRoot_ENDPOINT_REPLAY_PASS": bool(
            complete and all(row["endpoint_gate"] for row in rows)
        ),
        "state_count": sum(row["state_count"] for row in rows),
        "valid_state_count": sum(row["valid_state_count"] for row in rows),
        "mutation_count": sum(row["mutation_count"] for row in rows),
        "attempted_mutation_count": sum(
            row["attempted_mutation_count"] for row in rows
        ),
        "transition_counts": dict(sorted(transition_counts.items())),
        "trajectories_with_negative_transition": sum(
            row["transition_counts"].get("negative", 0) > 0 for row in rows
        ),
        "trajectories_with_invalid_intermediate": sum(
            row["valid_state_count"] < row["state_count"] for row in rows
        ),
        "tasks": rows,
    }
    output = args.run_dir / "aggregate_summary.json"
    output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
