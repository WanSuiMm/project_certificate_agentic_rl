#!/usr/bin/env python3
"""Verify q-qualified tasks reproduce their official target failure on Modal."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from swesmith_modal_executor import SWESmithModalExecutor


def check(executor: SWESmithModalExecutor, task: dict) -> dict:
    try:
        result = executor.call(task, mode="init")
        return {"instance_id": task["instance_id"], "status": "gold_valid" if result["gold_self_consistent"]
                else "gold_invalid", "clean_tests": result["clean_tests"],
                "buggy_f2p": result["buggy_f2p"],
                "buggy_source_sha256": result["buggy_source_sha256"],
                "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        return {"instance_id": task["instance_id"], "status": "infrastructure_error",
                "reason": type(exc).__name__, "detail": str(exc)[:500],
                "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--q-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("invalid worker count")
    q_ids = {row["instance_id"] for line in args.q_results.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line)) and row["status"] == "q_valid"}
    tasks = [row for line in args.tasks.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line)) and row["instance_id"] in q_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {row["instance_id"] for line in args.output.read_text(encoding="utf-8").splitlines()
                if (row := json.loads(line))} if args.output.exists() else set()
    pending = [task for task in tasks if task["instance_id"] not in previous]
    executor = SWESmithModalExecutor()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a", encoding="utf-8") as output:
        futures = {pool.submit(check, executor, task): task for task in pending}
        for future in as_completed(futures):
            result = future.result()
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps({"instance_id": result["instance_id"], "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
