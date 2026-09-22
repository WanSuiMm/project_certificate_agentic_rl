#!/usr/bin/env python3
"""Resumeable remote orchestration for every exact-bound selected trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], *, log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
    if check and completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-4000:]}")
    return completed


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def image_inventory(udocker: list[str]) -> str:
    return run([*udocker, "images"]).stdout


def loaded_ref(inventory: str, image_name: str) -> str | None:
    for line in inventory.splitlines():
        if image_name in line:
            return line.split()[0]
    return None


def acquire_image(
    *, image_name: str, inventory: str, downloads: Path, crane: Path, udocker: list[str], log_dir: Path
) -> tuple[str, str]:
    existing = loaded_ref(inventory, image_name)
    if existing:
        return existing, inventory
    downloads.mkdir(parents=True, exist_ok=True)
    tar_path = downloads / f"{slug(image_name)}_legacy.tar"
    candidates = [
        f"docker.1panel.live/{image_name}:latest",
        f"docker.1ms.run/{image_name}:latest",
        f"docker.io/{image_name}:latest",
    ]
    if not tar_path.exists():
        errors = []
        for ref in candidates:
            partial = tar_path.with_suffix(".partial")
            if partial.exists():
                partial.unlink()
            completed = run(
                [str(crane), "pull", "--platform", "linux/amd64", "--format", "legacy", ref, str(partial)],
                log=log_dir / f"image_pull_{slug(image_name)}_{len(errors)}.log",
                check=False,
            )
            if completed.returncode == 0:
                partial.replace(tar_path)
                break
            errors.append(completed.stdout[-2000:])
        else:
            raise RuntimeError(f"all image mirrors failed for {image_name}: {errors}")
    run([*udocker, "load", "-i", str(tar_path)], log=log_dir / f"image_load_{slug(image_name)}.log")
    inventory = image_inventory(udocker)
    ref = loaded_ref(inventory, image_name)
    if not ref:
        raise RuntimeError(f"loaded image not found in udocker inventory: {image_name}")
    return ref, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reuse-moto-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "batch_logs"
    failures = output / "failed_attempts"
    udocker = [sys.executable, str(root / "tools/udocker-1.3.17/udocker/maincmd.py")]
    crane = root / "tools/go-containerregistry-v0.22.1/crane"
    trajectories_path = root / "runs/selection_v01/selected_trajectories.jsonl"
    bindings_path = root / "runs/selection_v01/task_bindings.jsonl"
    trajectories = {row["instance_id"]: row for row in load_rows(trajectories_path)}
    bindings = [row for row in load_rows(bindings_path) if row["binding_status"] == "exact"]
    if len(bindings) != 25:
        raise RuntimeError(f"expected 25 exact bindings, found {len(bindings)}")

    if args.reuse_moto_dir:
        for source in args.reuse_moto_dir.glob("getmoto__*/"):
            target = output / source.name
            if (source / "summary.json").exists() and not target.exists():
                shutil.copytree(source, target)

    inventory = image_inventory(udocker)
    status = {
        "protocol": "exact25_intermediate_v01",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "expected_tasks": len(bindings),
        "completed": [],
        "failed": [],
    }
    state_path = output / "batch_state.json"

    for position, binding in enumerate(bindings, start=1):
        instance_id = binding["trajectory_instance_id"]
        row = trajectories[instance_id]
        traj_id = row["traj_id"]
        task_dir = output / instance_id
        if (task_dir / "summary.json").exists():
            status["completed"].append(traj_id)
            state_path.write_text(json.dumps(status, indent=2) + "\n")
            continue
        if task_dir.exists():
            failures.mkdir(parents=True, exist_ok=True)
            task_dir.replace(failures / f"{slug(instance_id)}_{position:02d}")
        image_name = binding["official_task"]["image_name"]
        try:
            ref, inventory = acquire_image(
                image_name=image_name,
                inventory=inventory,
                downloads=root / "downloads",
                crane=crane,
                udocker=udocker,
                log_dir=logs,
            )
            container = f"exact25_{position:02d}_{hashlib.sha256(traj_id.encode()).hexdigest()[:8]}"
            created = run([*udocker, "create", f"--name={container}", ref], check=False)
            if created.returncode and "already exists" not in created.stdout.lower():
                raise RuntimeError(created.stdout[-4000:])
            run([*udocker, "setup", "--execmode=P2", container])
            command = [
                *udocker,
                "run",
                f"--volume={root}:/audit",
                container,
                "bash",
                "-lc",
                " ".join(
                    [
                        "python /audit/scripts/run_moto_intermediate.py",
                        "--trajectories /audit/runs/selection_v01/selected_trajectories.jsonl",
                        "--bindings /audit/runs/selection_v01/task_bindings.jsonl",
                        f"--traj-id {traj_id}",
                        "--root /testbed",
                        f"--output-dir /audit/runs/exact25_intermediate_v01/{instance_id}",
                        "--pytest /opt/miniconda3/envs/testbed/bin/pytest",
                        "--timeout-seconds 1200",
                    ]
                ),
            ]
            completed = run(command, log=logs / f"task_{position:02d}_{slug(instance_id)}.log", check=False)
            if completed.returncode or not (task_dir / "summary.json").exists():
                raise RuntimeError(completed.stdout[-4000:])
            status["completed"].append(traj_id)
        except Exception as exc:
            status["failed"].append({"traj_id": traj_id, "error": str(exc)[-4000:]})
        state_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["complete"] = len(status["completed"]) == len(bindings)
    state_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
