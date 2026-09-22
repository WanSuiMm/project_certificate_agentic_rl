"""Select a frozen, balanced SWE-smith tool-trajectory audit panel.

This script does not replay repositories or infer semantic edit quality. It only
uses native tool-call metadata to locate explicit editor boundaries and writes a
compact, provenance-pinned panel for the later container replay audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "SWE-bench/SWE-smith-trajectories"
DEFAULT_REVISION = "08e109b4a59eaeebf80e4675cd125d42e7ac99a4"
EDITOR_COMMANDS = {"create", "insert", "str_replace"}


def parse_messages(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(messages, list):
        raise ValueError("messages must decode to a list")
    return messages


def extract_structured_edits(
    raw: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for message_index, message in enumerate(parse_messages(raw)):
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "str_replace_editor":
                continue
            arguments = function.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                continue
            command = parsed.get("command")
            if command not in EDITOR_COMMANDS:
                continue
            edits.append(
                {
                    "message_index": message_index,
                    "tool_call_id": tool_call.get("id"),
                    "command": command,
                    "path": parsed.get("path"),
                }
            )
    return edits


def repo_key(instance_id: str) -> str:
    return instance_id.split(".", 1)[0]


def stable_row_id(row: dict[str, Any]) -> str:
    payload = f"{row.get('instance_id', '')}\0{row.get('traj_id', '')}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_panel(
    rows: Iterable[dict[str, Any]],
    *,
    target_per_outcome: int,
    min_edits: int,
    max_per_repo: int,
    min_repos: int,
    max_scanned: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    outcome_counts: Counter[bool] = Counter()
    repo_counts: Counter[str] = Counter()
    scanned = 0

    for row in rows:
        scanned += 1
        if scanned > max_scanned:
            break
        outcome = bool(row["resolved"])
        if outcome_counts[outcome] >= target_per_outcome:
            continue
        repository = repo_key(row["instance_id"])
        if repo_counts[repository] >= max_per_repo:
            continue
        edits = extract_structured_edits(row["messages"])
        if len(edits) < min_edits:
            continue
        record = dict(row)
        record["selection_sha256"] = stable_row_id(row)
        record["repository"] = repository
        record["structured_edits"] = edits
        record["structured_edit_count"] = len(edits)
        selected.append(record)
        outcome_counts[outcome] += 1
        repo_counts[repository] += 1
        if (
            outcome_counts[True] >= target_per_outcome
            and outcome_counts[False] >= target_per_outcome
            and len(repo_counts) >= min_repos
        ):
            break

    summary = {
        "scanned_rows": scanned,
        "selected_rows": len(selected),
        "resolved": outcome_counts[True],
        "unresolved": outcome_counts[False],
        "repositories": dict(sorted(repo_counts.items())),
        "repository_count": len(repo_counts),
        "structured_edit_count": sum(r["structured_edit_count"] for r in selected),
    }
    if outcome_counts[True] < target_per_outcome:
        raise RuntimeError(f"only selected {outcome_counts[True]} resolved trajectories")
    if outcome_counts[False] < target_per_outcome:
        raise RuntimeError(f"only selected {outcome_counts[False]} unresolved trajectories")
    if len(repo_counts) < min_repos:
        raise RuntimeError(f"only selected {len(repo_counts)} repositories")
    return selected, summary


def write_panel(
    output_dir: Path,
    selected: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "selected_trajectories.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["selected_trajectories_sha256"] = hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="tool")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--target-per-outcome", type=int, default=20)
    parser.add_argument("--min-edits", type=int, default=2)
    parser.add_argument("--max-per-repo", type=int, default=5)
    parser.add_argument("--min-repos", type=int, default=8)
    parser.add_argument("--max-scanned", type=int, default=10000)
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset(
        args.dataset,
        revision=args.revision,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    selected, summary = select_panel(
        rows,
        target_per_outcome=args.target_per_outcome,
        min_edits=args.min_edits,
        max_per_repo=args.max_per_repo,
        min_repos=args.min_repos,
        max_scanned=args.max_scanned,
    )
    manifest = {
        "protocol": "swe_smith_frozen_trajectory_audit_v01",
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "target_per_outcome": args.target_per_outcome,
        "min_structured_edits": args.min_edits,
        "max_per_repository": args.max_per_repo,
        "min_repositories": args.min_repos,
        "max_scanned": args.max_scanned,
        "edit_boundary_definition": {
            "tool": "str_replace_editor",
            "commands": sorted(EDITOR_COMMANDS),
            "claim": "explicit structured editor calls only; not all filesystem mutations",
        },
        "summary": summary,
    }
    write_panel(args.output, selected, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
