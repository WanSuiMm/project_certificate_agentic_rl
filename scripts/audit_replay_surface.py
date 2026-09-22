#!/usr/bin/env python3
"""Audit undo and suspicious shell mutation exposure in the exact task stratum."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SHELL_TOOLS = {"bash", "shell", "terminal"}
SUSPICIOUS_PATTERNS = {
    "sed_in_place": re.compile(r"\bsed\s+(?:[^\n]*\s)?-i(?:\s|['\"]|$)", re.I),
    "perl_in_place": re.compile(r"\bperl\s+[^\n]*(?:-pi|-p\s+-i)\b", re.I),
    "python_write_text": re.compile(r"\bwrite_text\s*\(", re.I),
    "python_open_write": re.compile(
        r"\bopen\s*\([^\n]{0,160},\s*['\"](?:w|a|x)[+b]?['\"]", re.I
    ),
    "shell_redirect": re.compile(r"\b(?:cat|echo|printf)\b[^\n]*(?:>>|>)", re.I),
    "tee": re.compile(r"(?:^|[;&|]\s*)tee(?:\s|$)", re.I),
    "filesystem_command": re.compile(r"(?:^|[;&|]\s*)(?:mv|cp|rm|touch)\s", re.I),
    "git_mutation": re.compile(
        r"\bgit\s+(?:checkout|restore|apply|reset|clean)\b", re.I
    ),
    "patch_command": re.compile(r"(?:^|[;&|]\s*)patch(?:\s|$)", re.I),
}


def decode_messages(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, list):
        raise ValueError("messages must decode to a list")
    return value


def decode_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def audit_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    undo_calls: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    shell_call_count = 0
    for message_index, message in enumerate(decode_messages(row["messages"])):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            arguments = decode_arguments(function.get("arguments"))
            if name == "str_replace_editor" and arguments.get("command") == "undo_edit":
                undo_calls.append(
                    {
                        "message_index": message_index,
                        "tool_call_id": call.get("id"),
                        "path": arguments.get("path"),
                    }
                )
            if name not in SHELL_TOOLS:
                continue
            shell_call_count += 1
            command = str(arguments.get("command", ""))
            matches = sorted(
                label for label, pattern in SUSPICIOUS_PATTERNS.items() if pattern.search(command)
            )
            if matches:
                suspicious.append(
                    {
                        "message_index": message_index,
                        "tool_call_id": call.get("id"),
                        "patterns": matches,
                        "command": command,
                    }
                )
    return {
        "traj_id": row["traj_id"],
        "instance_id": row["instance_id"],
        "resolved": bool(row["resolved"]),
        "undo_edit_count": len(undo_calls),
        "undo_edit_calls": undo_calls,
        "shell_call_count": shell_call_count,
        "suspicious_shell_mutation_count": len(suspicious),
        "suspicious_shell_mutations": suspicious,
    }


def load_exact_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {
            row["trajectory_instance_id"]
            for row in map(json.loads, handle)
            if row["binding_status"] == "exact"
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    exact_ids = load_exact_ids(args.bindings)
    with args.trajectories.open("r", encoding="utf-8") as handle:
        rows = [
            row
            for row in map(json.loads, handle)
            if row["instance_id"] in exact_ids
        ]
    if len(rows) != len(exact_ids):
        raise RuntimeError(
            f"expected {len(exact_ids)} exact trajectories, found {len(rows)}"
        )

    audited = [audit_trajectory(row) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "replay_surface_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pattern_counts: Counter[str] = Counter()
    for row in audited:
        for call in row["suspicious_shell_mutations"]:
            pattern_counts.update(call["patterns"])
    summary = {
        "exact_trajectory_count": len(audited),
        "trajectory_selection_sha256": hashlib.sha256(
            args.trajectories.read_bytes()
        ).hexdigest(),
        "task_bindings_sha256": hashlib.sha256(args.bindings.read_bytes()).hexdigest(),
        "undo_edit_call_count": sum(row["undo_edit_count"] for row in audited),
        "trajectories_with_undo_edit": sum(
            bool(row["undo_edit_count"]) for row in audited
        ),
        "shell_call_count": sum(row["shell_call_count"] for row in audited),
        "suspicious_shell_mutation_call_count": sum(
            row["suspicious_shell_mutation_count"] for row in audited
        ),
        "trajectories_with_suspicious_shell_mutation": sum(
            bool(row["suspicious_shell_mutation_count"]) for row in audited
        ),
        "suspicious_pattern_counts": dict(sorted(pattern_counts.items())),
        "scope": "heuristic exposure audit only; not full shell replay",
    }
    (args.output_dir / "replay_surface_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
