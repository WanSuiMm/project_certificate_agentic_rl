#!/usr/bin/env python3
"""Qualify an outcome-blind SWE-smith pool with official-image q probes.

Writes each completed result immediately. A crash can be resumed with the same
output directory; only unprocessed task IDs are submitted again.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def check(task: dict, app, image) -> dict:
    import modal

    sandbox = modal.Sandbox.create(
        "python", "/opt/swesmith_q_qualifier_worker.py", app=app,
        image=image, timeout=240, cpu=1, memory=2048, block_network=True,
    )
    try:
        sandbox.stdin.write(json.dumps(task, ensure_ascii=False).encode())
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        sandbox.wait()
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        if sandbox.returncode != 0:
            result = {"status": "infrastructure_error", "returncode": sandbox.returncode,
                      "stderr_tail": stderr[-1000:], "stdout_tail": stdout[-1000:]}
        else:
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                result = {"status": "infrastructure_error", "reason": "invalid_worker_json",
                          "stdout_tail": stdout[-1000:]}
    except Exception as exc:
        result = {"status": "infrastructure_error", "reason": type(exc).__name__,
                  "detail": str(exc)[:500]}
    finally:
        sandbox.terminate()
        sandbox.detach()
    result["instance_id"] = task["instance_id"]
    result["image_name"] = task["image_name"]
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-valid", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=311)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--previous-results", type=Path,
                        help="Recheck only q-valid or generically remediable prior skips")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.target_valid:
        raise ValueError("invalid worker or target count")
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line]
    priority = ("string2string", "python-string-similarity", "flashtext", "thefuzz",
                "python-slugify", "r1chardj0n3s_1776_parse", "furl", "langdetect",
                "python-markdownify", "markupsafe", "textfsm", "parsimonious",
                "sqlglot", "python-hyper_1776_h11", "tomli")
    tasks.sort(key=lambda task: (next((i for i, name in enumerate(priority)
                                      if name in task["image_name"]), len(priority)),
                                 task["instance_id"]))
    tasks = tasks[:args.max_candidates]
    if args.previous_results:
        prior = {row["instance_id"]: row for line in args.previous_results.read_text(encoding="utf-8").splitlines()
                 if (row := json.loads(line))}
        retry_reasons = {"unknown_parameter_type", "variadic_or_keyword_only_signature",
                         "q_outside_0.05_to_0.95_headroom"}
        tasks = [task for task in tasks if prior.get(task["instance_id"], {}).get("status") == "q_valid"
                 or prior.get(task["instance_id"], {}).get("reason") in retry_reasons]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.jsonl"
    previous = ([json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
                if result_path.exists() else [])
    seen = {row["instance_id"] for row in previous}
    valid_count = sum(row["status"] == "q_valid" for row in previous)
    import modal

    app = modal.App.lookup("certificate-swesmith-q-probe-v01", create_if_missing=True)
    worker = Path(__file__).with_name("swesmith_q_qualifier_worker.py")
    images = {task["image_name"]: modal.Image.from_registry(task["image_name"]).add_local_file(
        str(worker), remote_path="/opt/swesmith_q_qualifier_worker.py") for task in tasks}
    pending = [task for task in tasks if task["instance_id"] not in seen]
    with ThreadPoolExecutor(max_workers=args.workers) as executor, result_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.workers):
            if valid_count >= args.target_valid:
                break
            chunk = pending[start:start + args.workers]
            futures = {executor.submit(check, task, app, images[task["image_name"]]): task for task in chunk}
            for future in as_completed(futures):
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                previous.append(result)
                valid_count += result["status"] == "q_valid"
                print(json.dumps({"checked": len(previous), "valid": valid_count,
                                  "status": result["status"], "instance_id": result["instance_id"],
                                  "reason": result.get("reason")}, ensure_ascii=False), flush=True)
    summary = {
        "status": "q_qualification_only_not_RL", "pool_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(), "target_valid": args.target_valid,
        "max_candidates": args.max_candidates, "workers": args.workers,
        "checked": len(previous), "valid": valid_count,
        "status_counts": dict(Counter(row["status"] for row in previous)),
        "skip_reasons": dict(Counter(row.get("reason") for row in previous if row["status"] == "skip")),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
