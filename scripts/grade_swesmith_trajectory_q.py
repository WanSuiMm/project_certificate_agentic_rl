#!/usr/bin/env python3
"""Offline q measurements for P0, P4, P8 after all trajectories finish."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from swesmith_modal_executor import SWESmithModalExecutor
from train_swesmith_agent_grpo import append, load_selected


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--q-results", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    trajectory_receipt = json.loads((args.trajectories / "run.json").read_text(encoding="utf-8"))
    if trajectory_receipt["status"] != "complete" or trajectory_receipt["states"] != 3612:
        raise RuntimeError("all 28 x 16 trajectories must reach P8 before offline q")
    state_path = args.trajectories / "states.jsonl"
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_states = {}
    for line in state_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "q" in row or "q_invalid_candidate" in row:
            raise RuntimeError("online trajectory contains q")
        if row["step"] not in {0, 4, 8}:
            continue
        key = row["instance_id"], row["trajectory_id"], row["step"]
        if key in selected_states or sha256(row["source"]) != row["source_sha256"]:
            raise RuntimeError(f"duplicate or corrupt checkpoint: {key}")
        selected_states[key] = row
    expected = {(task_id, None, 0) for task_id in selected["ids"]}
    expected |= {(task_id, sample, step) for task_id in selected["ids"]
                 for sample in range(16) for step in (4, 8)}
    if selected_states.keys() != expected or len(selected_states) != 924:
        raise RuntimeError("expected exactly 28 shared P0 and 448 each of P4/P8")

    states_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    metadata = {"kind": "offline_reference_q_P0_P4_P8", "status": "running",
                "states_sha256": states_sha, "requested_state_count": 924,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "run.json"
    measure_path = args.output_dir / "measurements.jsonl"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old["states_sha256"] != states_sha or old["kind"] != metadata["kind"]:
            raise RuntimeError("cannot resume q with changed trajectories")
        if old["status"] == "complete":
            return
        metadata["started_utc"] = old["started_utc"]
    else:
        receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    measured = {}
    if measure_path.exists():
        for line in measure_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["instance_id"], row["source_sha256"]
            if key in measured:
                raise RuntimeError(f"duplicate q measurement: {key}")
            measured[key] = row
    items = load_selected(args.tasks, args.q_results, selected["ids"])
    by_task = {item["task"]["instance_id"]: item for item in items}
    executor = SWESmithModalExecutor()
    for key in sorted(expected, key=lambda k: (k[0], k[2], -1 if k[1] is None else k[1])):
        row = selected_states[key]
        measure_key = row["instance_id"], row["source_sha256"]
        if measure_key in measured:
            continue
        item = by_task[row["instance_id"]]
        result = executor.call(item["task"], mode="proxy_only", source=row["source"],
                               buggy_source_sha256=selected_states[(row["instance_id"], None, 0)]["source_sha256"],
                               q_bank=item["q_bank"], include_proxy=True)
        if result["status"] not in {"q_scored", "q_invalid_candidate"}:
            raise RuntimeError(f"unexpected q result: {result['status']}")
        measurement = {"instance_id": row["instance_id"],
                       "source_sha256": row["source_sha256"],
                       "status": result["status"], "q": result["q"]}
        if "reason" in result:
            measurement["reason"] = result["reason"]
        append(measure_path, measurement)
        measured[measure_key] = measurement
    observation_path = args.output_dir / "observations.jsonl"
    with observation_path.open("w", encoding="utf-8") as handle:
        for key in sorted(expected, key=lambda k: (k[0], k[2], -1 if k[1] is None else k[1])):
            state = selected_states[key]
            result = measured[(state["instance_id"], state["source_sha256"])]
            handle.write(json.dumps({"instance_id": state["instance_id"],
                                     "trajectory_id": state["trajectory_id"], "step": state["step"],
                                     "source_sha256": state["source_sha256"],
                                     "status": result["status"], "q": result["q"]},
                                    ensure_ascii=False) + "\n")
    if len(observation_path.read_text(encoding="utf-8").splitlines()) != 924:
        raise RuntimeError("offline q observations incomplete")
    metadata.update(status="complete", unique_measured_states=len(measured),
                    valid_q=sum(row["status"] == "q_scored" for row in measured.values()),
                    finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"offline q complete: {len(measured)} unique states, 924 observations", flush=True)


if __name__ == "__main__":
    main()
