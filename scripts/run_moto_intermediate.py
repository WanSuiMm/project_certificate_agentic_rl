#!/usr/bin/env python3
"""Run one exact Moto trajectory and record tests at every mutation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from replay_structured_edits import apply_call, mutation_events, select_row


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_binding(path: Path, instance_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        matches = [
            row
            for row in map(json.loads, handle)
            if row["trajectory_instance_id"] == instance_id
        ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one task binding, found {len(matches)}")
    binding = matches[0]
    if binding["binding_status"] != "exact":
        raise RuntimeError(f"task is not exact-bound: {binding['binding_status']}")
    return binding


def git(root: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def initialize_task(root: Path, patch: str) -> dict[str, Any]:
    check = git(root, "apply", "--check", "-", input_text=patch)
    if check.returncode != 0:
        raise RuntimeError(f"task patch check failed: {check.stderr}")
    apply = git(root, "apply", "-", input_text=patch)
    if apply.returncode != 0:
        raise RuntimeError(f"task patch apply failed: {apply.stderr}")
    return {
        "patch_sha256": sha256_bytes(patch.encode()),
        "git_apply_check_returncode": check.returncode,
        "git_apply_returncode": apply.returncode,
    }


def workspace_snapshot(root: Path) -> dict[str, Any]:
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise RuntimeError(status.stderr)
    entries: list[dict[str, Any]] = []
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        relative = line[3:]
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        path = root / relative
        if path.is_symlink():
            digest = sha256_bytes(path.readlink().as_posix().encode())
            kind = "symlink"
        elif path.is_file():
            digest = sha256_bytes(path.read_bytes())
            kind = "file"
        elif path.is_dir():
            digest = None
            kind = "directory"
        else:
            digest = None
            kind = "missing"
        entries.append(
            {"status": code, "path": relative, "kind": kind, "sha256": digest}
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"entries": entries, "workspace_sha256": sha256_bytes(payload)}


def parse_junit(path: Path, requested: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - failed - errors - skipped
    return {
        "requested": requested,
        "collected": len(cases),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "nonpassing": failed + errors,
        "valid": len(cases) == requested and skipped == 0,
    }


def run_test_group(
    *,
    root: Path,
    pytest_path: Path,
    tests: list[str],
    state_dir: Path,
    label: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    junit = state_dir / f"{label}.xml"
    output = state_dir / f"{label}.txt"
    command = [
        str(pytest_path),
        "-q",
        "--tb=short",
        f"--junitxml={junit}",
        *tests,
    ]
    started = time.time()
    try:
        process = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.time() - started
        output.write_text(process.stdout + process.stderr, encoding="utf-8")
        if not junit.exists():
            return {
                "valid": False,
                "reason": "missing_junit",
                "returncode": process.returncode,
                "duration_seconds": duration,
            }
        result = parse_junit(junit, len(tests))
        result.update(
            {"returncode": process.returncode, "duration_seconds": duration}
        )
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        output.write_text(stdout + stderr, encoding="utf-8")
        return {
            "valid": False,
            "reason": "timeout",
            "returncode": None,
            "duration_seconds": duration,
        }


def transition_label(before: dict[str, Any], after: dict[str, Any]) -> str:
    if not before["valid"] or not after["valid"]:
        return "invalid"
    f0, r0 = before["F"], before["R"]
    f1, r1 = after["F"], after["R"]
    if f1 == f0 and r1 == r0:
        return "neutral"
    if f1 <= f0 and r1 <= r0:
        return "positive"
    if f1 >= f0 and r1 >= r0:
        return "negative"
    return "mixed"


def evaluate_state(
    *,
    state_index: int,
    root: Path,
    output_dir: Path,
    pytest_path: Path,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    preceding_edit: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    state_dir = output_dir / f"state_{state_index:03d}"
    state_dir.mkdir(parents=True, exist_ok=False)
    before_tests = workspace_snapshot(root)
    f2p = run_test_group(
        root=root,
        pytest_path=pytest_path,
        tests=fail_to_pass,
        state_dir=state_dir,
        label="fail_to_pass",
        timeout_seconds=timeout_seconds,
    )
    p2p = run_test_group(
        root=root,
        pytest_path=pytest_path,
        tests=pass_to_pass,
        state_dir=state_dir,
        label="pass_to_pass",
        timeout_seconds=timeout_seconds,
    )
    after_tests = workspace_snapshot(root)
    drift = before_tests["workspace_sha256"] != after_tests["workspace_sha256"]
    valid = bool(f2p.get("valid")) and bool(p2p.get("valid")) and not drift
    state = {
        "state_index": state_index,
        "preceding_edit": preceding_edit,
        "workspace_before_tests": before_tests,
        "workspace_after_tests": after_tests,
        "test_induced_workspace_drift": drift,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "F": f2p.get("nonpassing") if f2p.get("valid") else None,
        "R": p2p.get("nonpassing") if p2p.get("valid") else None,
        "valid": valid,
    }
    (state_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return state


def mutation_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in mutation_events(row) if event["succeeded"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--traj-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pytest", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    row = select_row(args.trajectories, args.traj_id)
    binding = load_binding(args.bindings, row["instance_id"])
    task = binding["official_task"]
    initialization = initialize_task(args.root, task["patch"])

    states: list[dict[str, Any]] = []
    states.append(
        evaluate_state(
            state_index=0,
            root=args.root,
            output_dir=args.output_dir,
            pytest_path=args.pytest,
            fail_to_pass=task["FAIL_TO_PASS"],
            pass_to_pass=task["PASS_TO_PASS"],
            preceding_edit=None,
            timeout_seconds=args.timeout_seconds,
        )
    )

    events = mutation_events(row)
    successful_calls = [event for event in events if event["succeeded"]]
    for index, call in enumerate(successful_calls, start=1):
        receipt = apply_call(args.root, call["arguments"])
        if receipt is None:
            raise RuntimeError(f"mutation unexpectedly ignored: {call}")
        receipt.update(
            {
                "message_index": call["message_index"],
                "tool_call_id": call["tool_call_id"],
            }
        )
        states.append(
            evaluate_state(
                state_index=index,
                root=args.root,
                output_dir=args.output_dir,
                pytest_path=args.pytest,
                fail_to_pass=task["FAIL_TO_PASS"],
                pass_to_pass=task["PASS_TO_PASS"],
                preceding_edit=receipt,
                timeout_seconds=args.timeout_seconds,
            )
        )

    transitions = [
        {
            "from": before["state_index"],
            "to": after["state_index"],
            "label": transition_label(before, after),
        }
        for before, after in zip(states, states[1:])
    ]
    final = states[-1]
    observed_resolved = bool(
        final["valid"] and final["F"] == 0 and final["R"] == 0
    )
    summary = {
        "protocol": "moto_intermediate_v01",
        "traj_id": row["traj_id"],
        "instance_id": row["instance_id"],
        "declared_resolved": bool(row["resolved"]),
        "observed_resolved": observed_resolved,
        "endpoint_agrees": observed_resolved == bool(row["resolved"]),
        "initialization": initialization,
        "state_count": len(states),
        "mutation_count": len(states) - 1,
        "attempted_mutation_count": len(events),
        "unsuccessful_mutation_attempts": [
            {
                "message_index": event["message_index"],
                "tool_call_id": event["tool_call_id"],
                "command": event["arguments"].get("command"),
                "tool_response_first_line": event["tool_response_first_line"],
            }
            for event in events
            if not event["succeeded"]
        ],
        "all_states_valid": all(state["valid"] for state in states),
        "curve": [
            {
                "state_index": state["state_index"],
                "F": state["F"],
                "R": state["R"],
                "valid": state["valid"],
            }
            for state in states
        ],
        "transitions": transitions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
