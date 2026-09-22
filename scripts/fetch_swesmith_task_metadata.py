#!/usr/bin/env python3
"""Bind a frozen trajectory selection to official SWE-smith task metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset


DATASET = "SWE-bench/SWE-smith-py"
REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["instance_id"]] = row
    if not rows:
        raise ValueError(f"no instance_id values found in {path}")
    return rows


def profile_id(instance_id: str) -> str:
    return instance_id.rsplit(".", 1)[0]


def failing_tests(trajectory: dict) -> list[str]:
    messages = trajectory["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )
    match = re.search(r"<failing_test>\s*(.*?)\s*</failing_test>", user_text, re.S)
    return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    trajectories = selected_rows(args.selection)
    wanted = set(trajectories)
    wanted_profiles = {profile_id(instance_id) for instance_id in wanted}
    dataset = load_dataset(
        DATASET,
        split="train",
        revision=REVISION,
        streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    matched: dict[str, dict] = {}
    profile_refs: dict[str, dict] = {}
    scanned = 0
    for row in dataset:
        scanned += 1
        instance_id = row["instance_id"]
        profile = profile_id(instance_id)
        if profile in wanted_profiles and profile not in profile_refs:
            profile_refs[profile] = dict(row)
        if instance_id in wanted:
            matched[instance_id] = dict(row)
        if len(matched) == len(wanted) and len(profile_refs) == len(wanted_profiles):
                break

    missing = sorted(wanted - matched.keys())
    missing_profiles = sorted(
        {profile_id(instance_id) for instance_id in missing} - profile_refs.keys()
    )
    if missing_profiles:
        raise RuntimeError(
            f"missing {len(missing_profiles)} selected repo profiles: {missing_profiles[:5]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "task_bindings.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for instance_id in sorted(wanted):
            if instance_id in matched:
                binding = {
                    "binding_status": "exact",
                    "trajectory_instance_id": instance_id,
                    "official_task": matched[instance_id],
                }
            else:
                ref = profile_refs[profile_id(instance_id)]
                binding = {
                    "binding_status": "profile_fallback",
                    "trajectory_instance_id": instance_id,
                    "repo_profile": profile_id(instance_id),
                    "image_name": ref["image_name"],
                    "reference_instance_id": ref["instance_id"],
                    "failing_tests_from_trajectory_prompt": failing_tests(
                        trajectories[instance_id]
                    ),
                    "limitation": "exact task absent from pinned current dataset revision",
                }
            handle.write(json.dumps(binding, ensure_ascii=False) + "\n")

    image_counts: dict[str, int] = {}
    for instance_id in wanted:
        row = matched.get(instance_id, profile_refs[profile_id(instance_id)])
        image_counts[row["image_name"]] = image_counts.get(row["image_name"], 0) + 1

    manifest = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "selection_sha256": sha256(args.selection),
        "task_bindings_sha256": sha256(output),
        "selected_task_count": len(wanted),
        "exact_task_count": len(matched),
        "profile_fallback_count": len(missing),
        "unique_image_count": len(image_counts),
        "image_counts": dict(sorted(image_counts.items())),
        "dataset_rows_scanned": scanned,
    }
    manifest_path = args.output_dir / "task_bindings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
