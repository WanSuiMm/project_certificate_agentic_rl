#!/usr/bin/env python3
"""Select a reproducible, image-reuse-aware candidate slice of official SWE-smith tasks.

This only freezes official task rows. It does not establish that the images run
on Modal, that gold patches pass, or that a semantic proxy is available.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


DATASET = "SWE-bench/SWE-smith-py"
REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"
SEED = "swesmith-agentic-64-v01"


def digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode()).hexdigest()


def changed_paths(patch: str) -> list[str]:
    paths = []
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            pair = line.split(" b/", 1)
            if len(pair) != 2:
                return []
            paths.append(pair[1])
    return paths


def eligible(row: dict) -> bool:
    patch = row.get("patch") or ""
    paths = changed_paths(patch)
    if len(paths) != 1 or not paths[0].endswith(".py"):
        return False
    parts = paths[0].lower().split("/")
    if any(part in {"test", "tests", "testing"} or part.startswith("test_") for part in parts):
        return False
    f2p = row.get("FAIL_TO_PASS") or []
    p2p = row.get("PASS_TO_PASS") or []
    return bool(
        row.get("instance_id") and row.get("image_name")
        and (row.get("problem_statement") or "").strip()
        and 1 <= len(f2p) <= 10 and 1 <= len(p2p) <= 100
        and len(patch) <= 6000
    )


def select(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        if eligible(row) and row["instance_id"] not in seen:
            groups[row["image_name"]].append(row)
            seen.add(row["instance_id"])
    viable = [image for image, tasks in groups.items() if len(tasks) >= 8]
    images = sorted(viable, key=digest)[:8]
    if len(images) != 8:
        raise ValueError(f"need 8 images with 8 eligible tasks each; found {len(images)}")
    selected = []
    for image in images:
        tasks = sorted(groups[image], key=lambda row: digest(row["instance_id"]))[:8]
        for index, row in enumerate(tasks):
            selected.append({
                "split": "train" if index < 6 else "heldout",
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "image_name": image,
                "problem_statement": row["problem_statement"],
                "patch": row["patch"],
                "FAIL_TO_PASS": row["FAIL_TO_PASS"],
                "PASS_TO_PASS": row["PASS_TO_PASS"],
            })
    summary = {
        "eligible_rows": sum(map(len, groups.values())),
        "viable_images": len(viable),
        "selected_images": images,
        "split_counts": dict(Counter(row["split"] for row in selected)),
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET, split="train", revision=REVISION, streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    rows_scanned = 0
    candidates = []
    for row in dataset:
        rows_scanned += 1
        if eligible(row):
            candidates.append(row)
    selected, summary = select(candidates)
    args.output_dir.mkdir(parents=True)
    tasks_path = args.output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "official_task_candidates_not_runtime_qualified",
        "dataset": DATASET, "revision": REVISION, "seed": SEED,
        "rows_scanned": rows_scanned, "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "selection": "single non-test Python file, nonempty issue, <=6000 patch chars, "
                     "1-10 F2P, 1-100 P2P; 8 hash-ranked images x 8 hash-ranked tasks; "
                     "per image 6 train / 2 heldout",
        **summary,
        "not_verified": ["Modal image startup", "gold reverse patch", "candidate agent loop",
                         "semantic proxy", "GRPO training"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
