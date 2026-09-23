#!/usr/bin/env python3
"""Wait for the frozen census, then run public-only continuation and offline q."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    write_receipt(receipt, "waiting_for_census", census=str(args.census),
                  trajectories=str(args.trajectories), offline_q=str(args.offline_q))
    try:
        while True:
            census_receipt = args.census / "run.json"
            if census_receipt.exists():
                status = json.loads(census_receipt.read_text(encoding="utf-8"))["status"]
                if status == "complete":
                    break
                if status != "running":
                    raise RuntimeError(f"census ended without complete status: {status}")
            try:
                os.kill(args.census_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("census process exited without complete receipt") from exc
            time.sleep(60)
        write_receipt(receipt, "continuing_trajectories")
        subprocess.run([sys.executable, str(scripts / "continue_swesmith_body_trajectories.py"),
                        "--config", str(args.config), *common, "--census", str(args.census),
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
