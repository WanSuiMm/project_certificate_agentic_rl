#!/usr/bin/env python3
"""Wait for a capture-only P0..P8 run, then score q offline exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def record(path: Path, status: str, **fields) -> None:
    path.write_text(json.dumps({
        "status": status, "time_utc": datetime.now(timezone.utc).isoformat(),
        **fields}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "tasks", "q-results", "trajectories", "output-dir", "receipt-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--capture-pid", required=True, type=int)
    args = parser.parse_args()
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt_dir / "run.json"
    record(receipt_path, "waiting_for_capture", capture_pid=args.capture_pid)
    try:
        while True:
            capture_receipt = args.trajectories / "run.json"
            if capture_receipt.exists():
                status = json.loads(capture_receipt.read_text(encoding="utf-8"))["status"]
                if status == "complete":
                    break
                if status != "running":
                    raise RuntimeError(f"capture ended with status {status}")
            try:
                os.kill(args.capture_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("capture exited without complete receipt") from exc
            time.sleep(30)
        record(receipt_path, "grading_q_offline")
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("grade_swesmith_trajectory_q.py")),
            "--selection", str(args.selection), "--tasks", str(args.tasks),
            "--q-results", str(args.q_results), "--trajectories", str(args.trajectories),
            "--output-dir", str(args.output_dir)], check=True)
    except Exception as exc:
        record(receipt_path, "failed", reason=type(exc).__name__, detail=str(exc)[:500])
        raise
    record(receipt_path, "complete", observations=3612,
           q_output=str(args.output_dir))


if __name__ == "__main__":
    main()
