#!/usr/bin/env python3
"""Run one q-compatibility check in an official SWE-smith Modal image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    task = next((json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines()
                 if line and json.loads(line)["instance_id"] == args.instance_id), None)
    if task is None:
        raise ValueError("instance not found")
    import modal

    app = modal.App.lookup("certificate-swesmith-q-probe-v01", create_if_missing=True)
    worker = Path(__file__).with_name("swesmith_q_qualifier_worker.py")
    image = modal.Image.from_registry(task["image_name"]).add_local_file(
        str(worker), remote_path="/opt/swesmith_q_qualifier_worker.py"
    )
    sandbox = modal.Sandbox.create(
        "python", "/opt/swesmith_q_qualifier_worker.py", app=app, image=image,
        timeout=240, cpu=1, memory=2048, block_network=True,
    )
    try:
        sandbox.stdin.write(json.dumps(task, ensure_ascii=False).encode("utf-8"))
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        sandbox.wait()
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        if sandbox.returncode != 0:
            result = {"status": "infrastructure_error", "returncode": sandbox.returncode,
                      "stderr_tail": stderr[-3000:], "stdout_tail": stdout[-3000:]}
        else:
            result = json.loads(stdout)
    finally:
        sandbox.terminate()
        sandbox.detach()
    result["image_name"] = task["image_name"]
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "reference"}},
                     ensure_ascii=False, indent=2))
    if result["status"] == "infrastructure_error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
