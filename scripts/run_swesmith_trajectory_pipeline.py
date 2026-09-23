#!/usr/bin/env python3
"""Stream ready P1 blocks into public-only trajectories, then grade all states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path: Path, status: str, **fields) -> None:
    path.write_text(json.dumps({"status": status, "time_utc": utc_now(), **fields},
                               indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "census", "trajectories",
                 "offline-q", "receipt-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--census-pid", required=True, type=int)
    args = parser.parse_args()
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.receipt_dir / "run.json"
    scripts = Path(__file__).resolve().parent
    common = ["--selection", str(args.selection), "--tasks", str(args.tasks),
              "--q-results", str(args.q_results)]
    write_receipt(receipt, "continuing_trajectories", census=str(args.census),
                  trajectories=str(args.trajectories), offline_q=str(args.offline_q))
    try:
        subprocess.run([sys.executable, str(scripts / "continue_swesmith_body_trajectories.py"),
                        "--config", str(args.config), *common, "--census", str(args.census),
                        "--census-pid", str(args.census_pid),
                        "--output-dir", str(args.trajectories)], check=True)
        write_receipt(receipt, "grading_q_offline")
        subprocess.run([sys.executable, str(scripts / "grade_swesmith_trajectory_q.py"),
                        *common, "--trajectories", str(args.trajectories),
                        "--output-dir", str(args.offline_q)], check=True)
    except Exception as exc:
        write_receipt(receipt, "failed", reason=type(exc).__name__, detail=str(exc)[:500])
        raise
    write_receipt(receipt, "complete", trajectories=str(args.trajectories),
                  offline_q=str(args.offline_q))


if __name__ == "__main__":
    main()
