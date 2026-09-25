#!/usr/bin/env python3
"""Score frozen Oracle Credit P2 sources with the original reference q bank.

This is a post-hoc measurement only: no actor generation, public tests, or
optimizer updates. A saved P1 q is remeasured in each task image before P2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(run_dir: Path, tasks_path: Path, q_path: Path) -> tuple[dict, dict, dict, dict]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete" or run["completed_edit_steps"] != run["expected_edit_steps"]:
        raise ValueError("Oracle Credit source run is not complete")
    if digest(tasks_path) != run["tasks_sha256"] or digest(q_path) != run["q_results_sha256"]:
        raise ValueError("task metadata or frozen q bank differs from the source run")
    task_ids = run["task_ids"]
    tasks = {x["instance_id"]: x for x in rows(tasks_path) if x["instance_id"] in task_ids}
    banks = {x["instance_id"]: x for x in rows(q_path)
             if x["instance_id"] in task_ids and x["status"] == "q_valid"}
    if set(tasks) != set(task_ids) or set(banks) != set(task_ids):
        raise ValueError("missing selected task or valid frozen q bank")
    if any(len(banks[t]["cases"]) != 256 or len(banks[t]["reference"]) != 256 for t in task_ids):
        raise ValueError("frozen q bank is not 256 cases per task")

    candidates = {}
    p1_by_source = {}
    buggy_hashes = {}
    for row in rows(run_dir / "candidates.jsonl"):
        task_id, sample = row["instance_id"], row["sample"]
        key = task_id, sample
        if key in candidates or source_hash(row["source"]) != row["source_sha256"]:
            raise ValueError(f"duplicate or corrupt P1 candidate: {key}")
        candidates[key] = row
        if row["invalid_body"]:
            buggy_hashes.setdefault(task_id, set()).add(row["source_sha256"])
        if row["q"] is not None:
            source_key = task_id, row["source_sha256"]
            if source_key in p1_by_source and p1_by_source[source_key] != row["q"]:
                raise ValueError(f"inconsistent P1 q for identical source: {source_key}")
            p1_by_source[source_key] = row["q"]
    expected_candidates = len(task_ids) * run["candidates_per_task"]
    if len(candidates) != expected_candidates or any(len(buggy_hashes.get(t, ())) != 1 for t in task_ids):
        raise ValueError("P1 candidate panel or buggy-source anchors are incomplete")

    p2 = {}
    for row in rows(run_dir / "states.jsonl"):
        if row["step"] != 2:
            continue
        key = row["instance_id"], row["sample"], row["replicate"]
        if (key in p2 or key[:2] not in candidates or source_hash(row["source"]) != row["source_sha256"]
                or row["prior_source_sha256"] != candidates[key[:2]]["source_sha256"]):
            raise ValueError(f"duplicate, corrupt or mislinked P2 state: {key}")
        p2[key] = row
    expected_p2 = expected_candidates * run["continuations_per_candidate"]
    if len(p2) != expected_p2:
        raise ValueError(f"P2 states missing: {len(p2)} != {expected_p2}")
    return run, tasks, banks, {"candidates": candidates, "p1_by_source": p1_by_source,
                               "buggy_hashes": buggy_hashes, "p2": p2}


def run_grade(args: argparse.Namespace, executor_factory=None) -> dict:
    run_dir, output_dir = args.run_dir.resolve(strict=True), args.output_dir
    run, tasks, banks, data = load_inputs(run_dir, args.tasks, args.q_results)
    task_ids = run["task_ids"]
    unique_p2 = {(r["instance_id"], r["source_sha256"]): r for r in data["p2"].values()}
    pinned = {"source_run_sha256": digest(run_dir / "run.json"),
              "states_sha256": digest(run_dir / "states.jsonl"),
              "candidates_sha256": digest(run_dir / "candidates.jsonl"),
              "tasks_sha256": run["tasks_sha256"], "q_results_sha256": run["q_results_sha256"]}
    receipt_path = output_dir / "run.json"
    if output_dir.exists() and not receipt_path.exists():
        raise FileExistsError("output directory exists without a resumable receipt")
    output_dir.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["kind"] != "oracle_credit_p2_frozen_q" or receipt["pinned"] != pinned:
            raise ValueError("cannot resume P2 q with changed inputs")
        if receipt["status"] == "complete":
            return receipt
    else:
        receipt = {"kind": "oracle_credit_p2_frozen_q", "status": "running",
                   "task_ids": task_ids, "expected_p2_observations": len(data["p2"]),
                   "unique_p2_sources": len(unique_p2), "pinned": pinned,
                   "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(receipt_path, receipt)

    journal = output_dir / "measurements.jsonl"
    measured = {}
    if journal.exists():
        for row in rows(journal):
            key = row["instance_id"], row["source_sha256"]
            if key in measured or key not in unique_p2 or row["status"] not in {"q_scored", "q_invalid_candidate"}:
                raise ValueError(f"invalid saved P2 q measurement: {key}")
            measured[key] = row
    pending = [key for task_id in task_ids for key in sorted(unique_p2)
               if key[0] == task_id and key not in measured and key not in data["p1_by_source"]]
    if pending:
        if executor_factory is None:
            sys.path.insert(0, str(args.source_scripts.resolve(strict=True)))
            from swesmith_modal_executor import SWESmithModalExecutor
            executor_factory = SWESmithModalExecutor
        executor = executor_factory()
        new_count = 0
        for task_id in task_ids:
            task_keys = [key for key in pending if key[0] == task_id]
            if not task_keys:
                continue
            if args.max_new is not None and new_count >= args.max_new:
                break
            with executor.task_session(tasks[task_id]) as session:
                initialized = session.initialize(mode="source_only")
                if initialized["buggy_source_sha256"] not in data["buggy_hashes"][task_id]:
                    raise ValueError(f"official image has a different injected P0: {task_id}")
                session.set_q_bank(banks[task_id])
                controls = [r for (t, _), r in data["candidates"].items()
                            if t == task_id and r["q"] is not None and not r["invalid_body"]]
                control = max(controls, key=lambda r: r["q"])
                check = session.call(mode="proxy_only", source=control["source"])
                if check["status"] != "q_scored" or check["q"] != control["q"]:
                    raise ValueError(f"saved P1 q fails image/bank parity: {task_id}")
                for key in task_keys:
                    if args.max_new is not None and new_count >= args.max_new:
                        break
                    state = unique_p2[key]
                    result = session.call(mode="proxy_only", source=state["source"])
                    if result["status"] not in {"q_scored", "q_invalid_candidate"}:
                        raise RuntimeError(f"unexpected q result: {result['status']}")
                    row = {"instance_id": task_id, "source_sha256": key[1],
                           "status": result["status"], "q": result["q"],
                           "reason": result.get("reason"), "origin": "observer"}
                    append(journal, row)
                    measured[key] = row
                    new_count += 1
                    print(f"P2 q {len(measured)}/{len(pending)} measured; task={task_id}; new={new_count}", flush=True)

    if all(key in measured or key in data["p1_by_source"] for key in unique_p2):
        observations = output_dir / "observations.jsonl"
        with observations.open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(data["p2"]):
                state = data["p2"][key]
                source_key = key[0], state["source_sha256"]
                existing = measured.get(source_key)
                q = existing["q"] if existing is not None else data["p1_by_source"][source_key]
                handle.write(json.dumps({"instance_id": key[0], "sample": key[1],
                                         "replicate": key[2], "step": 2,
                                         "source_sha256": source_key[1], "p2": state["p_T"],
                                         "solved_p2": state["solved"], "q2": q,
                                         "q_origin": "observer" if existing else "same_source_as_P1"},
                                        ensure_ascii=False, sort_keys=True) + "\n")
        receipt.update(status="complete", unique_new_measured=len(measured),
                       reused_p1_sources=len(unique_p2) - len(measured),
                       valid_q_observations=sum(x["q2"] is not None for x in rows(observations)),
                       finished_utc=datetime.now(timezone.utc).isoformat())
    else:
        receipt.update(status="partial", unique_new_measured=len(measured))
    write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "tasks", "q-results", "source-scripts", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if args.max_new is not None and args.max_new < 1:
        parser.error("--max-new must be positive")
    print(json.dumps(run_grade(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
