#!/usr/bin/env python3
"""Build an outcome-blind SWE-smith pool for automatic q qualification.

The pool deliberately prefers small parser/string/protocol repositories. It
does not use F2P/P2P contents to generate probes or select by policy outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from freeze_swesmith_agentic_candidates import DATASET, REVISION, eligible


IMAGE_KEYWORDS = (
    "python-string-similarity", "tomli", "python-hyper_1776_h11",
    "sqlglot", "parsimonious", "string2string", "furl", "langdetect",
    "python-markdownify", "r1chardj0n3s_1776_parse", "flashtext",
    "thefuzz", "python-slugify", "markupsafe", "textfsm",
)
SEED = "swesmith-qfirst-pool-v02"


def family(instance_id: str) -> str:
    return instance_id.rsplit(".", 1)[-1].split("__", 1)[0]


def pool_eligible(row: dict) -> bool:
    name = family(row.get("instance_id", ""))
    return eligible(row) and (name == "func_basic" or name.startswith("func_pm_")
                              or name == "lm_rewrite") and any(
        keyword in row["image_name"] for keyword in IMAGE_KEYWORDS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-per-image", type=int, default=80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from datasets import load_dataset

    stream = load_dataset(DATASET, split="train", revision=REVISION, streaming=True,
                          cache_dir=str(args.cache_dir) if args.cache_dir else None)
    groups: dict[str, list[dict]] = defaultdict(list)
    scanned = 0
    for row in stream:
        scanned += 1
        if pool_eligible(row):
            groups[row["image_name"]].append(row)
    rank = lambda row: hashlib.sha256(f"{SEED}\0{row['instance_id']}".encode()).hexdigest()
    chosen = [row for image in sorted(groups) for row in sorted(groups[image], key=rank)[:args.max_per_image]]
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "tasks.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "q_qualification_pool_not_policy_training_set",
        "dataset": DATASET, "revision": REVISION, "seed": SEED,
        "image_keywords": IMAGE_KEYWORDS, "max_per_image": args.max_per_image,
        "rows_scanned": scanned, "eligible_counts": dict(sorted((k, len(v)) for k, v in groups.items())),
        "chosen_count": len(chosen), "families": dict(Counter(family(r["instance_id"]) for r in chosen)),
        "tasks_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
