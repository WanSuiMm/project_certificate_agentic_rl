#!/usr/bin/env python3
"""Check real-image bug injection, official tests, and frozen q scorer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from swesmith_modal_executor import SWESmithModalExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--q-results", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    task = next((json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines()
                 if line and json.loads(line)["instance_id"] == args.instance_id), None)
    q = next((json.loads(line) for line in args.q_results.read_text(encoding="utf-8").splitlines()
              if line and json.loads(line)["instance_id"] == args.instance_id
              and json.loads(line)["status"] == "q_valid"), None)
    if task is None or q is None:
        raise ValueError("task or q bank not found")
    executor = SWESmithModalExecutor()
    initialized = executor.call(task, mode="init")
    if not initialized["gold_self_consistent"]:
        raise RuntimeError(f"official-image gold self-consistency failed: {initialized}")
    scored = executor.call(task, mode="score", source=initialized["buggy_source"],
                           buggy_source_sha256=initialized["buggy_source_sha256"],
                           q_bank=q, include_proxy=True)
    if scored["solved"] or abs(scored["q"] - q["q_initial"]) > 1e-9:
        raise RuntimeError(f"buggy endpoint scorer inconsistent with q qualification: {scored}")
    receipt = {"status": "smoke_passed", "instance_id": args.instance_id,
               "gold_self_consistent": True, "buggy_solved": scored["solved"],
               "p_initial": scored["p_T"], "q_initial": scored["q"],
               "q_bank_sha256": q["bank_sha256"],
               "worker_sha256": hashlib.sha256(Path(__file__).with_name("swesmith_agent_worker.py").read_bytes()).hexdigest(),
               "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
