"""Freeze and validate task files and identical per-arm training schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from function_swe import TaskError, load_task


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(task_dir: Path, output: Path, *, seed: int = 0) -> dict[str, Any]:
    paths = sorted(task_dir.glob("*.json"))
    if len(paths) != 64:
        raise TaskError(f"expected exactly 64 task JSON files, got {len(paths)}")
    tasks = [(path, load_task(path)) for path in paths]
    ids = [task["task_id"] for _, task in tasks]
    if len(set(ids)) != 64:
        raise TaskError("duplicate task_id")
    random.Random(seed).shuffle(tasks)
    entries = []
    for index, (path, task) in enumerate(tasks):
        try:
            relative_path = path.resolve().relative_to(output.parent.resolve())
        except ValueError as exc:
            raise TaskError("task directory must be inside the manifest directory") from exc
        entries.append({
            "task_id": task["task_id"],
            "path": relative_path.as_posix(),
            "sha256": file_sha256(path),
            "split": "train" if index < 48 else "heldout",
        })
    manifest = {"protocol": "oracle_proxy_grpo_survival_v01", "seed": seed, "tasks": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "oracle_proxy_grpo_survival_v01":
        raise TaskError("wrong manifest protocol")
    entries = manifest.get("tasks", [])
    if len(entries) != 64 or [e["split"] for e in entries].count("train") != 48 or [e["split"] for e in entries].count("heldout") != 16:
        raise TaskError("manifest must contain 48 train and 16 held-out tasks")
    if len({e["task_id"] for e in entries}) != 64:
        raise TaskError("duplicate task IDs")
    root = path.parent.resolve()
    tasks = []
    for entry in entries:
        task_path = (root / entry["path"]).resolve()
        if not task_path.is_relative_to(root):
            raise TaskError("task path escapes manifest directory")
        if file_sha256(task_path) != entry["sha256"]:
            raise TaskError(f"task file changed: {entry['task_id']}")
        task = load_task(task_path)
        if task["task_id"] != entry["task_id"]:
            raise TaskError("manifest/task ID mismatch")
        tasks.append({**task, "split": entry["split"]})
    return manifest, tasks


def task_schedule(train_ids: list[str], *, seed: int, updates: int, tasks_per_update: int) -> list[list[str]]:
    if len(train_ids) < tasks_per_update or len(set(train_ids)) != len(train_ids):
        raise ValueError("training IDs must be unique and fill an update")
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < updates:
        cycle = train_ids.copy()
        rng.shuffle(cycle)
        for start in range(0, len(cycle), tasks_per_update):
            group = cycle[start : start + tasks_per_update]
            if len(group) == tasks_per_update:
                schedule.append(group)
            if len(schedule) == updates:
                break
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = freeze_manifest(args.task_dir, args.output, seed=args.seed)
    print(f"frozen {len(result['tasks'])} tasks at {args.output}")


if __name__ == "__main__":
    main()
