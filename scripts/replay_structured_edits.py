#!/usr/bin/env python3
"""Replay explicit str_replace_editor mutations with fail-closed semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row["messages"]
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, list):
        raise ValueError("messages must decode to a list")
    return value


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def mutation_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return attempted editor mutations with their observed tool outcome.

    A tool call is a filesystem transition only when the linked tool observation
    confirms success. Failed replacement attempts remain in the receipt but are
    not replayed as states.
    """
    decoded = messages(row)
    responses: dict[str, str] = {}
    for message in decoded:
        if message.get("role") != "tool":
            continue
        response_text = content_text(message.get("content"))
        for call_id in message.get("tool_call_ids") or []:
            responses[call_id] = response_text

    events: list[dict[str, Any]] = []
    for message_index, message in enumerate(decoded):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "str_replace_editor":
                continue
            raw = function.get("arguments") or "{}"
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            command = arguments.get("command")
            if command not in {"create", "insert", "str_replace"}:
                continue
            call_id = call.get("id")
            response = responses.get(call_id, "")
            succeeded = (
                "File created successfully at:" in response
                if command == "create"
                else " has been edited." in response
            )
            events.append(
                {
                    "message_index": message_index,
                    "tool_call_id": call_id,
                    "arguments": arguments,
                    "succeeded": succeeded,
                    "tool_response_first_line": next(
                        (
                            line
                            for line in response.splitlines()
                            if line and line != "OBSERVATION:"
                        ),
                        "",
                    )[:500],
                }
            )
    return events


def select_row(path: Path, traj_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        matches = [json.loads(line) for line in handle if line.strip() and traj_id in line]
    exact = [row for row in matches if row.get("traj_id") == traj_id]
    if len(exact) != 1:
        raise RuntimeError(f"expected one exact traj_id match, found {len(exact)}")
    return exact[0]


def confined_path(root: Path, tool_path: str) -> Path:
    prefix = "/testbed/"
    if not tool_path.startswith(prefix):
        raise ValueError(f"path is outside /testbed: {tool_path}")
    result = (root / tool_path[len(prefix) :]).resolve()
    root_resolved = root.resolve()
    if result != root_resolved and root_resolved not in result.parents:
        raise ValueError(f"resolved path escapes root: {tool_path}")
    return result


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def apply_call(root: Path, arguments: dict[str, Any]) -> dict[str, Any] | None:
    command = arguments.get("command")
    if command not in {"create", "insert", "str_replace"}:
        return None
    path = confined_path(root, arguments["path"])
    before = digest(path)

    if command == "create":
        if path.exists():
            raise RuntimeError(f"create target already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["file_text"], encoding="utf-8")
    elif command == "str_replace":
        old = arguments["old_str"]
        new = arguments["new_str"]
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"str_replace expected one match in {path}, found {count}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    else:
        line = int(arguments["insert_line"])
        original = path.read_text(encoding="utf-8")
        parts = original.splitlines(keepends=True)
        if not 0 <= line <= len(parts):
            raise RuntimeError(f"insert line {line} outside 0..{len(parts)} for {path}")
        insertion = arguments["new_str"]
        if insertion and not insertion.endswith("\n"):
            insertion += "\n"
        parts.insert(line, insertion)
        path.write_text("".join(parts), encoding="utf-8")

    return {
        "command": command,
        "tool_path": arguments["path"],
        "sha256_before": before,
        "sha256_after": digest(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--traj-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    row = select_row(args.trajectories, args.traj_id)
    applied: list[dict[str, Any]] = []
    events = mutation_events(row)
    for event in events:
        if not event["succeeded"]:
            continue
        result = apply_call(args.root, event["arguments"])
        if result is not None:
            result.update(
                {
                    "message_index": event["message_index"],
                    "tool_call_id": event["tool_call_id"],
                }
            )
            applied.append(result)

    receipt = {
        "traj_id": args.traj_id,
        "instance_id": row["instance_id"],
        "declared_resolved": bool(row["resolved"]),
        "top_level_patch_sha256": hashlib.sha256(row["patch"].encode()).hexdigest(),
        "applied_edit_count": len(applied),
        "attempted_edit_count": len(events),
        "unsuccessful_edit_attempts": [
            {
                "message_index": event["message_index"],
                "tool_call_id": event["tool_call_id"],
                "command": event["arguments"].get("command"),
                "tool_response_first_line": event["tool_response_first_line"],
            }
            for event in events
            if not event["succeeded"]
        ],
        "applied_edits": applied,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
