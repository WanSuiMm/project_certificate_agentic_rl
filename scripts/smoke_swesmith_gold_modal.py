#!/usr/bin/env python3
"""Verify one official SWE-smith task in its real image on Modal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    rows = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line]
    task = next((row for row in rows if row["instance_id"] == args.instance_id), None)
    if task is None:
        raise ValueError("instance not in frozen candidate file")
    import modal

    app = modal.App.lookup("certificate-swesmith-gold-smoke-v01", create_if_missing=True)
    worker = Path(__file__).with_name("swesmith_gold_worker.py")
    image = modal.Image.from_registry(task["image_name"]).add_local_file(
        str(worker), remote_path="/opt/swesmith_gold_worker.py"
    )
    sandbox = modal.Sandbox.create(
        "python", "/opt/swesmith_gold_worker.py", app=app, image=image,
        timeout=300, cpu=1, memory=2048, block_network=True,
    )
    try:
        sandbox.stdin.write(json.dumps(task, ensure_ascii=False).encode("utf-8"))
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        sandbox.wait()
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        receipt = {
            "image_name": task["image_name"], "instance_id": task["instance_id"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": sandbox.returncode,
            "result": json.loads(stdout) if sandbox.returncode == 0 else None,
            "stderr_tail": stderr[-2000:],
        }
    finally:
        sandbox.terminate()
        sandbox.detach()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["returncode"] != 0 or not receipt["result"]["self_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
