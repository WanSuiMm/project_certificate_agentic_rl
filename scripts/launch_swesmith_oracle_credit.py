#!/usr/bin/env python3
"""Launch the frozen oracle pilot in a new remote run directory; save receipt."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "source-dir", "legacy-root", "hf-home", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--import-run-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    source, legacy, output = args.source_dir.resolve(), args.legacy_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("launch requires a new run directory; resume with the saved command")
    # Preserve the venv executable symlink: resolving it selects the base env.
    command = [str(args.python.absolute()), "-u", str(source / "scripts/run_swesmith_oracle_credit.py"),
               "--config", str(source / "configs/swesmith_oracle_credit_6x16x4_v01.json"),
               "--selection", str(legacy / "runs/swesmith_survival_20x8_selection_v01.json"),
               "--census", str(legacy / "runs/body_census_1p5b_28x16_v01/results.jsonl"),
               "--tasks", str(legacy / "src/tasks.jsonl"),
               "--q-results", str(legacy / "runs/q_qualification_v03/results.jsonl"),
               "--output-dir", str(output)]
    for index in (0, 2, 4, 6, 8, 10, 12):
        if not Path(command[index]).is_file():
            raise FileNotFoundError(command[index])
    if args.import_run_dir is not None:
        source_run = args.import_run_dir.resolve(strict=True)
        if not (source_run / "run.json").is_file():
            raise FileNotFoundError(source_run / "run.json")
        command.extend(["--import-run-dir", str(source_run)])
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(HF_HOME=str(args.hf_home.resolve()), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=args.gpu,
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    log_path = output / "process.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    receipt = {"pid": process.pid, "host": socket.gethostname(), "gpu": args.gpu,
               "command": command, "cwd": str(source), "log": str(log_path),
               "output": str(output), "started_utc": datetime.now(timezone.utc).isoformat(),
               "environment_overrides": {key: env[key] for key in
                   ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES",
                    "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED")}}
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pid": process.pid, "log": str(log_path), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
