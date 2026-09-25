"""Export reviewable Oracle Credit evidence without source or server receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STATE_FIELDS = (
    "instance_id",
    "sample",
    "replicate",
    "step",
    "source_sha256",
    "prior_source_sha256",
    "invalid_body",
    "batch_seed",
    "generated_tokens",
    "hit_token_cap",
    "p_T",
    "solved",
    "public_status",
    "public_seconds",
)
OUTCOME_FIELDS = (
    "instance_id",
    "sample",
    "replicate",
    "solved",
    "p8",
    "source_sha256",
    "public_valid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_rows(source: Path, destination: Path, fields: tuple[str, ...]) -> int:
    count = 0
    with source.open(encoding="utf-8") as incoming, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as outgoing:
        for line in incoming:
            row = json.loads(line)
            missing = set(fields) - row.keys()
            if missing:
                raise ValueError(f"{source.name} row {count} lacks {sorted(missing)}")
            outgoing.write(json.dumps({key: row[key] for key in fields}, sort_keys=True) + "\n")
            count += 1
    return count


def export(run_dir: Path, output_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if run["status"] != "complete" or not summary["completeness"]["complete"]:
        raise ValueError("refusing to publish an incomplete run")
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = {
        "states": _export_rows(
            run_dir / "states.jsonl", output_dir / "states_public.jsonl", STATE_FIELDS
        ),
        "outcomes": _export_rows(
            run_dir / "outcomes.jsonl",
            output_dir / "outcomes_public.jsonl",
            OUTCOME_FIELDS,
        ),
    }
    if counts != {
        "states": run["expected_edit_steps"],
        "outcomes": run["expected_endpoints"],
    }:
        raise ValueError(f"row-count mismatch: {counts}")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol": run["config"]["protocol"],
        "status": run["status"],
        "run_started_utc": run["started_utc"],
        "run_finished_utc": run["finished_utc"],
        "model": run["config"]["model"],
        "model_revision": run["config"]["model_revision"],
        "config": run["config"],
        "source_input_sha256": {
            key: run[key]
            for key in (
                "selection_sha256",
                "census_sha256",
                "tasks_sha256",
                "q_results_sha256",
            )
        },
        "raw_artifact_sha256": {
            name: _sha256(run_dir / name)
            for name in ("run.json", "candidates.jsonl", "states.jsonl", "outcomes.jsonl", "summary.json")
        },
        "published_rows": counts,
        "omitted_fields": "Source text, model completions, issue text, test feedback text, logs, launch receipts, and machine identifiers remain in the private raw archive.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
