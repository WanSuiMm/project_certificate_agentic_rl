#!/usr/bin/env python3
"""Write a compact public index of a frozen SWE-smith candidate JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(tasks: Path, manifest: Path) -> list[dict]:
    raw = tasks.read_bytes()
    expected = json.loads(manifest.read_text(encoding="utf-8"))["tasks_sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("task JSONL does not match frozen manifest")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) != 64 or len({row["instance_id"] for row in rows}) != 64:
        raise ValueError("expected 64 unique task IDs")
    return [
        {
            "instance_id": row["instance_id"],
            "image_name": row["image_name"],
            "split": row["split"],
            "f2p_count": len(row["FAIL_TO_PASS"]),
            "p2p_count": len(row["PASS_TO_PASS"]),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = summarize(args.tasks, args.manifest)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"indexed {len(rows)} task IDs at {args.output}")


if __name__ == "__main__":
    main()
