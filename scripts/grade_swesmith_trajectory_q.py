#!/usr/bin/env python3
"""Offline q measurements for every P0..P8 state after trajectories finish."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from capture_swesmith_body_trajectories import validate_feedback_chain
from swesmith_modal_executor import SWESmithModalExecutor
from train_swesmith_agent_grpo import append, load_selected


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def validate_capture_receipt(receipt: dict, selection: dict,
                             selection_sha256: str) -> tuple[list[str], int, int, int]:
    task_ids = receipt.get("task_ids", [])
    count = receipt.get("trajectories_per_task")
    steps = receipt.get("steps")
    expected_states = (len(task_ids) * (1 + count * steps)
                       if isinstance(count, int) and isinstance(steps, int) else -1)
    if (receipt.get("kind") != "fresh_eight_step_closed_loop_public_feedback_q_offline"
            or receipt.get("status") != "complete"
            or not task_ids or task_ids != selection["ids"][:len(task_ids)]
            or count != 16 or steps != 8
            or receipt.get("selection_sha256") != selection_sha256
            or receipt.get("states") != expected_states):
        raise RuntimeError("offline q requires a complete fresh closed-loop capture")
    return task_ids, count, steps, expected_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--q-results", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    trajectory_receipt = json.loads((args.trajectories / "run.json").read_text(encoding="utf-8"))
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    task_ids, count, steps, expected_states = validate_capture_receipt(
        trajectory_receipt, selected, hashlib.sha256(args.selection.read_bytes()).hexdigest())
    state_path = args.trajectories / "states.jsonl"
    states_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if trajectory_receipt.get("states_sha256") != states_sha:
        raise RuntimeError("trajectory states changed after capture completion")
    selected_states = {}
    for line in state_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if ("q" in row or "q_invalid_candidate" in row
                or not all(field in row for field in ("public", "p_T", "solved"))):
            raise RuntimeError("online trajectory is missing public feedback or contains q")
        key = row["instance_id"], row["trajectory_id"], row["step"]
        if key in selected_states or sha256(row["source"]) != row["source_sha256"]:
            raise RuntimeError(f"duplicate or corrupt checkpoint: {key}")
        selected_states[key] = row
    expected = {(task_id, None, 0) for task_id in task_ids}
    expected |= {(task_id, sample, step) for task_id in task_ids
                 for sample in range(count) for step in range(1, steps + 1)}
    expected_observations = expected_states
    if selected_states.keys() != expected or len(selected_states) != expected_observations:
        raise RuntimeError("missing required P0..P8 checkpoints")
    validate_feedback_chain(selected_states)

    metadata = {"kind": "offline_reference_q_closed_loop_P0_P8", "status": "running",
                "task_ids": task_ids,
                "states_sha256": states_sha, "requested_state_count": expected_observations,
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
    items = load_selected(args.tasks, args.q_results, task_ids)
    by_task = {item["task"]["instance_id"]: item for item in items}
    executor = SWESmithModalExecutor()
    for task_id in task_ids:
        keys = sorted((key for key in expected if key[0] == task_id),
                      key=lambda k: (k[2], -1 if k[1] is None else k[1]))
        pending = [key for key in keys if (task_id, selected_states[key]["source_sha256"])
                   not in measured]
        if not pending:
            continue
        item = by_task[task_id]
        with executor.task_session(item["task"]) as session:
            initialized = session.initialize(mode="source_only")
            if initialized["buggy_source_sha256"] != selected_states[(task_id, None, 0)]["source_sha256"]:
                raise RuntimeError(f"P0 source changed before offline q: {task_id}")
            session.set_q_bank(item["q_bank"])
            for key in pending:
                row = selected_states[key]
                measure_key = task_id, row["source_sha256"]
                if measure_key in measured:
                    continue
                result = session.call(mode="proxy_only", source=row["source"])
                if result["status"] not in {"q_scored", "q_invalid_candidate"}:
                    raise RuntimeError(f"unexpected q result: {result['status']}")
                measurement = {"instance_id": task_id,
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
                                     "p_T": state["p_T"], "solved": state["solved"],
                                     "public": state["public"],
                                     "invalid_body": state.get("invalid_body", False),
                                     "status": result["status"], "q": result["q"]},
                                    ensure_ascii=False) + "\n")
    if len(observation_path.read_text(encoding="utf-8").splitlines()) != expected_observations:
        raise RuntimeError("offline q observations incomplete")
    metadata.update(status="complete", unique_measured_states=len(measured),
                    valid_q=sum(row["status"] == "q_scored" for row in measured.values()),
                    finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"offline q complete: {len(measured)} unique states, {expected_observations} observations", flush=True)


if __name__ == "__main__":
    main()
