#!/usr/bin/env python3
"""Start one official SWE-smith image in Modal and inspect its repo shell.

This does not grade a task or claim RL readiness. It is a bounded image gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--image-substring", default="python-hyper_1776_h11")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    rows = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line]
    matched = [row for row in rows if args.image_substring in row["image_name"]]
    if not matched:
        raise ValueError("selected image is absent from task file")
    image_name = matched[0]["image_name"]
    import modal

    app = modal.App.lookup("certificate-swesmith-image-smoke-v01", create_if_missing=True)
    image = modal.Image.from_registry(image_name)
    sandbox = modal.Sandbox.create(
        "bash", "-lc", "pwd; git rev-parse --show-toplevel; python --version",
        app=app, image=image, timeout=90, cpu=1, memory=2048,
        block_network=True,
    )
    try:
        sandbox.wait()
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        result = {
            "image_name": image_name,
            "sample_instance_id": matched[0]["instance_id"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": sandbox.returncode,
            "stdout": stdout[:4000],
            "stderr": stderr[:4000],
        }
    finally:
        sandbox.terminate()
        sandbox.detach()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["returncode"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
