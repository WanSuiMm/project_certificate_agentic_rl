#!/usr/bin/env python3
"""Sample and score 16 one-edit body actions on each frozen SWE-smith task.

This is a base-policy diagnostic, not a GRPO update or a heldout evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from itertools import combinations
from pathlib import Path
import random

import torch

from oracle_proxy_reward import terminal_reward
from function_swe import TaskError
from swesmith_modal_executor import SWESmithModalExecutor, ScorerError
from train_swesmith_agent_grpo import append, episode, load_policy, load_selected


def information_metrics(rows: list[dict], task_ids: list[str]) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["instance_id"]].append(row)
    result = {
        "expected_candidates": len(task_ids) * 16,
        "recorded_candidates": len(rows),
        "complete_tasks": 0,
        "syntactically_valid_actions": 0,
        "executable_candidates": 0,
        "test_informative_tasks_all": 0,
        "q_informative_tasks_all": 0,
        "test_informative_tasks_executable": 0,
        "q_informative_tasks_executable": 0,
        "same_test_pairs_executable": 0,
        "q_splits_test_tie_pairs_executable": 0,
        "tasks_with_q_split_test_ties_executable": 0,
    }
    for task_id in task_ids:
        task_rows = by_task[task_id]
        if len(task_rows) != 16 or {row["sample"] for row in task_rows} != set(range(16)):
            continue
        result["complete_tasks"] += 1
        scored = [row for row in task_rows if row.get("status") == "scored"]
        result["syntactically_valid_actions"] += sum(
            row.get("invalid_edits") == 0 for row in task_rows if "score" in row)
        result["executable_candidates"] += len(scored)
        for suffix, subset in (("all", [row for row in task_rows if "score" in row]),
                               ("executable", scored)):
            result[f"test_informative_tasks_{suffix}"] += len(
                {row["score"]["p_T"] for row in subset}) > 1
            result[f"q_informative_tasks_{suffix}"] += len(
                {row["score"]["q"] for row in subset}) > 1
        same_test = sum(a["score"]["p_T"] == b["score"]["p_T"]
                        for a, b in combinations(scored, 2))
        split = sum(a["score"]["p_T"] == b["score"]["p_T"]
                    and a["score"]["q"] != b["score"]["q"]
                    for a, b in combinations(scored, 2))
        result["same_test_pairs_executable"] += same_test
        result["q_splits_test_tie_pairs_executable"] += split
        result["tasks_with_q_split_test_ties_executable"] += split > 0
    denominator = result["expected_candidates"]
    result["syntactically_valid_action_rate"] = result["syntactically_valid_actions"] / denominator
    result["executable_candidate_rate"] = result["executable_candidates"] / denominator
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--q-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    if selected["status"] != "q_first_frozen" or selected["count"] != 28 or len(selected["ids"]) != 28:
        raise ValueError("expected the frozen 28-task selection")
    if config["max_edits"] != 1 or config["max_generated_tokens_per_edit"] != 1024:
        raise ValueError("expected one body edit with a 1024-token cap")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    items = load_selected(args.tasks, args.q_results, selected["ids"])
    tokenizer, model = load_policy(config, device)
    executor = SWESmithModalExecutor()
    args.output_dir.mkdir(parents=True)
    metadata = {
        "status": "running", "kind": "base_policy_body_sampling_not_rl",
        "tasks": 28, "samples_per_task": 16,
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": config["model"], "model_revision": config["model_revision"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    counts = {"scored": 0, "invalid_body": 0, "scorer_error": 0, "task_error": 0,
              "solved": 0}
    rows = []
    for item in items:
        task_id = item["task"]["instance_id"]
        split = selected["splits"][task_id]
        for sample in range(16):
            row = {"instance_id": task_id, "split": split, "sample": sample}
            try:
                result = episode(item, model, tokenizer, executor, config=config,
                                 device=device, sample=True, proxy=True)
            except ScorerError as exc:
                row.update(status="scorer_error", error=str(exc)[:500])
                counts["scorer_error"] += 1
                if "buggy_source" not in item:
                    append(args.output_dir / "results.jsonl", row)
                    rows.append(row)
                    break
            except (RuntimeError, TaskError) as exc:
                # Initialization failures are task-level, never low-reward candidates.
                if "buggy_source" in item:
                    raise
                row.update(status="task_error", error=str(exc)[:500])
                counts["task_error"] += 1
                append(args.output_dir / "results.jsonl", row)
                rows.append(row)
                break
            else:
                row.update(status=result.score["status"], score=result.score,
                           invalid_edits=result.invalid_edits,
                           final_source_sha256=result.final_source_sha256,
                           completion=result.actions[0].completion,
                           generated_tokens=(len(result.actions[0].token_ids)
                                             - result.actions[0].prompt_length))
                if result.score["status"] == "invalid_body":
                    counts["invalid_body"] += 1
                else:
                    counts["scored"] += 1
                counts["solved"] += int(result.score["solved"])
                row["test_reward"] = terminal_reward("test", solved=result.score["solved"],
                                                     public_pass_fraction=result.score["p_T"])
                row["semantic_reward"] = terminal_reward("semantic", solved=result.score["solved"],
                                                         reference_agreement=result.score["q"])
            append(args.output_dir / "results.jsonl", row)
            rows.append(row)
        print(f"{task_id} complete", flush=True)
    information = information_metrics(rows, selected["ids"])
    complete = (information["complete_tasks"] == 28 and counts["scorer_error"] == 0
                and counts["task_error"] == 0)
    metadata.update(status="complete" if complete else "incomplete",
                    counts=counts, information_metrics=information,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
