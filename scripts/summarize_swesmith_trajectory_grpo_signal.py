#!/usr/bin/env python3
"""Frozen-policy eight-step GRPO reward-resolution census (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pvariance


EPS = 1e-12


def first_change(values: list[float]) -> int | None:
    return next((index for index in range(1, len(values))
                 if abs(values[index] - values[0]) > EPS), None)


def group_summary(task_id: str, curves: list[list[dict]]) -> dict:
    if len(curves) != 16 or any(len(curve) != 9 for curve in curves):
        raise ValueError("expected 16 complete P0..P8 curves per task")
    endpoints = [curve[8] for curve in curves]
    terminal = [float(row["solved"]) for row in endpoints]
    test = [y + 0.5 * float(row["p_T"]) for y, row in zip(terminal, endpoints)]
    semantic = [y + 0.5 * float(row["q"]) for y, row in zip(terminal, endpoints)]
    vp, vq = pvariance(test), pvariance(semantic)
    early_q = 0
    silent_q = 0
    for curve in curves:
        ps = [float(row["p_T"]) for row in curve]
        qs = [float(row["q"]) for row in curve]
        tp, tq = first_change(ps), first_change(qs)
        early_q += tq is not None and (tp is None or tq < tp)
        silent_q += any(abs(ps[t] - ps[t - 1]) <= EPS and
                        curve[t]["solved"] == curve[t - 1]["solved"] and
                        abs(qs[t] - qs[t - 1]) > EPS for t in range(1, 9))
    return {
        "task_id": task_id, "trajectories": 16,
        "terminal_variance": pvariance(terminal),
        "test_reward_variance": vp,
        "semantic_reward_variance": vq,
        "all_terminal_fail": all(y == 0 for y in terminal),
        "test_group_tied": vp <= EPS,
        "semantic_group_informative": vq > EPS,
        "semantic_rescues_test_tie": vp <= EPS and vq > EPS,
        "q_changes_before_public_count": early_q,
        "silent_q_movement_trajectory_count": silent_q,
    }


def summarize(states: list[dict], observations: list[dict],
              task_ids: list[str]) -> dict:
    expected = {(task_id, None, 0) for task_id in task_ids}
    expected |= {(task_id, sample, step) for task_id in task_ids
                 for sample in range(16) for step in range(1, 9)}
    keyed_states = {(r["instance_id"], r["trajectory_id"], r["step"]): r for r in states}
    keyed_q = {(r["instance_id"], r["trajectory_id"], r["step"]): r for r in observations}
    if (len(states) != len(expected) or len(observations) != len(expected)
            or keyed_states.keys() != expected or keyed_q.keys() != expected):
        raise ValueError("incomplete or duplicated P0..P8 observations")
    joined = {}
    for key in expected:
        state, measured = keyed_states[key], keyed_q[key]
        if state["source_sha256"] != measured["source_sha256"]:
            raise ValueError(f"source/q hash mismatch: {key}")
        q = measured.get("q")
        if measured.get("status") != "q_scored" or not isinstance(q, (int, float)) or not 0 <= q <= 1:
            raise ValueError(f"missing valid q: {key}")
        if (state["p_T"] != measured["p_T"] or state["solved"] != measured["solved"]):
            raise ValueError(f"public observation changed after capture: {key}")
        joined[key] = {"p_T": state["p_T"], "solved": state["solved"], "q": q}
    per_task = []
    for task_id in task_ids:
        p0 = joined[(task_id, None, 0)]
        curves = [[p0] + [joined[(task_id, sample, step)] for step in range(1, 9)]
                  for sample in range(16)]
        per_task.append(group_summary(task_id, curves))
    return {
        "kind": "frozen_policy_trajectory_grpo_reward_resolution",
        "tasks": len(task_ids), "trajectories_per_task": 16, "steps": 8,
        "semantic_rescue_tasks": sum(row["semantic_rescues_test_tie"] for row in per_task),
        "all_terminal_fail_tasks": sum(row["all_terminal_fail"] for row in per_task),
        "test_informative_tasks": sum(not row["test_group_tied"] for row in per_task),
        "semantic_informative_tasks": sum(row["semantic_group_informative"] for row in per_task),
        "q_changes_before_public_trajectories": sum(row["q_changes_before_public_count"] for row in per_task),
        "silent_q_movement_trajectories": sum(row["silent_q_movement_trajectory_count"] for row in per_task),
        "per_task": per_task,
        "interpretation": "observational reward resolution only; not an RL result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--offline-q", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture = json.loads((args.trajectories / "run.json").read_text(encoding="utf-8"))
    q_receipt = json.loads((args.offline_q / "run.json").read_text(encoding="utf-8"))
    if (capture.get("status") != "complete" or q_receipt.get("status") != "complete"
            or capture.get("task_ids") != q_receipt.get("task_ids")
            or capture.get("states_sha256") != q_receipt.get("states_sha256")):
        raise RuntimeError("complete matching capture and q receipts required")
    states = [json.loads(line) for line in (args.trajectories / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    observations = [json.loads(line) for line in (args.offline_q / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    result = summarize(states, observations, capture["task_ids"])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
