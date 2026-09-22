#!/usr/bin/env python3
"""Audit alignment between trajectory tool edits and the top-level patch field."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DIFF_PATH = re.compile(r"^diff --git a/(.*?) b/", re.MULTILINE)
MUTATIONS = {"create", "insert", "str_replace"}


def parse_messages(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(raw) if isinstance(raw, str) else raw


def mutation_paths(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    all_paths: set[str] = set()
    non_create_paths: set[str] = set()
    for message in parse_messages(row["messages"]):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "str_replace_editor":
                continue
            raw = function.get("arguments") or "{}"
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            command = arguments.get("command")
            if command not in MUTATIONS or not arguments.get("path"):
                continue
            path = arguments["path"].removeprefix("/testbed/")
            all_paths.add(path)
            if command != "create":
                non_create_paths.add(path)
    return all_paths, non_create_paths


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    patch = row.get("patch") or ""
    patch_paths = set(DIFF_PATH.findall(patch))
    all_edits, non_create_edits = mutation_paths(row)
    non_create_overlap = patch_paths & non_create_edits
    all_overlap = patch_paths & all_edits
    if not patch:
        classification = "empty_patch"
    elif non_create_overlap:
        classification = "aligned_non_create_path"
    elif not non_create_edits:
        classification = "no_non_create_editor_mutation"
    else:
        classification = "disjoint_nonempty_patch"
    return {
        "traj_id": row["traj_id"],
        "instance_id": row["instance_id"],
        "resolved": bool(row["resolved"]),
        "classification": classification,
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "patch_bytes": len(patch.encode()),
        "patch_paths": sorted(patch_paths),
        "all_editor_mutation_paths": sorted(all_edits),
        "non_create_editor_mutation_paths": sorted(non_create_edits),
        "all_path_overlap": sorted(all_overlap),
        "non_create_path_overlap": sorted(non_create_overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.trajectories.open("r", encoding="utf-8") as handle:
        audited = [audit_row(json.loads(line)) for line in handle if line.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "patch_alignment_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audited:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    by_outcome: dict[str, dict[str, int]] = {}
    for outcome in (True, False):
        counts = Counter(
            row["classification"] for row in audited if row["resolved"] is outcome
        )
        by_outcome[str(outcome).lower()] = dict(sorted(counts.items()))

    resolved_with_non_create = [
        row
        for row in audited
        if row["resolved"] and row["non_create_editor_mutation_paths"]
    ]
    resolved_aligned = [
        row for row in resolved_with_non_create if row["non_create_path_overlap"]
    ]
    summary = {
        "trajectory_count": len(audited),
        "selection_sha256": hashlib.sha256(args.trajectories.read_bytes()).hexdigest(),
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in audited).items())
        ),
        "classification_counts_by_resolved": by_outcome,
        "resolved_with_non_create_editor_mutation": len(resolved_with_non_create),
        "resolved_with_non_create_path_overlap": len(resolved_aligned),
        "resolved_non_create_path_alignment_rate": (
            len(resolved_aligned) / len(resolved_with_non_create)
            if resolved_with_non_create
            else None
        ),
        "interpretation": (
            "Path alignment is a necessary integrity check, not proof that patch content "
            "is semantically correct. Disjoint or empty patches must not be replayed."
        ),
    }
    (args.output_dir / "patch_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
