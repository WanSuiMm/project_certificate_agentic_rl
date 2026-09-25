#!/usr/bin/env python3
"""Seal a stopped oracle run so its durable journal can be imported safely."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    if run["status"] != "running":
        raise RuntimeError(f"refusing to interrupt status={run['status']}")
    try:
        os.kill(launch["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("launch process is still alive")
    backup = run_dir / "run_before_interrupt.json"
    if backup.exists():
        raise RuntimeError("interruption backup already exists")
    shutil.copy2(run_path, backup)
    counts = {"completed_edit_steps": sum(1 for _ in (run_dir / "states.jsonl").open(encoding="utf-8")),
              "completed_endpoints": sum(1 for _ in (run_dir / "outcomes.jsonl").open(encoding="utf-8"))}
    run.update(status="interrupted", interruption_reason=args.reason,
               interrupted_utc=datetime.now(timezone.utc).isoformat(), **counts)
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_path)
    print(json.dumps({"status": run["status"], **counts}), flush=True)


if __name__ == "__main__":
    main()
