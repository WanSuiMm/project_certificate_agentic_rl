#!/usr/bin/env python3
"""Two trusted toy scores on real Modal Sandboxes; write a launch-gate receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from function_swe import sha256_text
from function_swe_manifest import file_sha256
from modal_function_swe_executor import ModalSandboxExecutor


def toy_task() -> dict:
    buggy = "def double(x):\n    return x + 1\n"
    expected = lambda x: {"kind": "return", "value": 2 * x}
    return {
        "task_id": "trusted-modal-smoke", "target_function": "double",
        "buggy_source": buggy, "buggy_sha256": sha256_text(buggy),
        "reference_sha256": sha256_text("def double(x):\n    return 2 * x\n"),
        "public_cases": [{"args": [2], "kwargs": {}, "expect": expected(2)}],
        "hidden_cases": [{"args": [3], "kwargs": {}, "expect": expected(3)}],
        "probe_cases": [
            {"args": [x], "kwargs": {}, "expect": expected(x)} for x in range(4, 260)
        ],
    }


def run_smoke(executor: ModalSandboxExecutor) -> None:
    task = toy_task()
    buggy_score = executor.score(task, task["buggy_source"], terminal=True)
    if buggy_score["solved"] or buggy_score["q"] != 0:
        raise RuntimeError("buggy fixture scored incorrectly")
    fixed = "def double(x):\n    return x * 2\n"
    fixed_score = executor.score(task, fixed, terminal=True)
    if not fixed_score["solved"] or fixed_score["p_T"] != 1 or fixed_score["q"] != 1:
        raise RuntimeError("fixed fixture scored incorrectly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    executor = ModalSandboxExecutor()
    run_smoke(executor)
    scripts = Path(__file__).resolve().parent
    receipt = {
        "executor": "modal_sandbox_v01", "isolated_code_execution": True,
        "smoke_passed": True, "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_name": "certificate-function-swe-v01", "block_network": True,
        "fresh_sandbox_per_score": True, "worker_timeout_seconds": executor.timeout_seconds,
        "function_swe_sha256": file_sha256(scripts / "function_swe.py"),
        "worker_sha256": file_sha256(scripts / "function_swe_worker.py"),
        "modal_executor_sha256": file_sha256(scripts / "modal_function_swe_executor.py"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Modal sandbox smoke passed; receipt: {args.receipt}")


if __name__ == "__main__":
    main()
