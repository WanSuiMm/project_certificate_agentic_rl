#!/usr/bin/env python3
"""Capture P0..P8 source trajectories without per-step tests or q scoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import torch

from continue_swesmith_body_trajectories import ready_task_census, save_state, sha256, state_key
from swesmith_agent_edit import InvalidEdit, current_callable_body, replace_callable_body
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import generate_action
from train_swesmith_agent_grpo import agent_prompt, load_policy, load_selected


SYSTEM_PROMPT = "You repair Python functions. Output only the replacement function body, without def or explanation."
NO_FEEDBACK = "No tests were run in this capture-only trajectory."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "census", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["status"] != "q_first_frozen" or selection["count"] != 28:
        raise ValueError("expected frozen 28-task selection")
    if config["max_edits"] != 1 or config["max_generated_tokens_per_edit"] != 1024:
        raise ValueError("unexpected action configuration")
    census_receipt = json.loads((args.census / "run.json").read_text(encoding="utf-8"))
    if census_receipt["status"] != "complete":
        raise RuntimeError("frozen P1 census must be complete before capture-only run")
    census_path = args.census / "results.jsonl"
    census_sha = hashlib.sha256(census_path.read_bytes()).hexdigest()
    task_ids = selection["ids"]
    frozen = {}
    for index, task_id in enumerate(task_ids):
        block = ready_task_census(args.census, task_id, index)
        if block is None:
            raise RuntimeError(f"missing frozen P1 block: {task_id}")
        for row in block:
            frozen[task_id, row["sample"]] = {key: row[key] for key in
                                               ("status", "completion", "final_source_sha256",
                                                "generated_tokens")}
    if len(frozen) != 448:
        raise RuntimeError("expected 448 frozen P1 actions")

    metadata = {"kind": "eight_step_capture_only_no_online_tests_or_q",
                "status": "running", "tasks": 28, "trajectories_per_task": 16,
                "steps": 8, "census_sha256": census_sha,
                "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "model_revision": config["model_revision"],
                "started_utc": datetime.now(timezone.utc).isoformat()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "run.json"
    states_path = args.output_dir / "states.jsonl"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if any(old.get(key) != metadata[key] for key in
               ("kind", "tasks", "trajectories_per_task", "steps", "census_sha256",
                "selection_sha256", "config_sha256", "model_revision")):
            raise RuntimeError("cannot resume capture with changed inputs")
        if old["status"] == "complete":
            return
        metadata["started_utc"] = old["started_utc"]
    else:
        receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    known = {}
    if states_path.exists():
        for line in states_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = state_key(row)
            if key in known or sha256(row["source"]) != row["source_sha256"]:
                raise RuntimeError(f"duplicate or corrupt saved state: {key}")
            if any(field in row for field in ("q", "p_T", "solved", "public")):
                raise RuntimeError(f"capture state contains a score: {key}")
            known[key] = row

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer, model = load_policy(config, device)
    executor = SWESmithModalExecutor()
    items = load_selected(args.tasks, args.q_results, task_ids)
    for item in items:
        task = item["task"]
        task_id = task["instance_id"]
        if all((task_id, sample, 8) in known for sample in range(16)):
            continue
        # One official-image source fetch applies the task mutation, but runs
        # neither clean/buggy tests nor per-state public or semantic scoring.
        init = executor.call(task, mode="source_only")
        if init["status"] != "source_only":
            raise RuntimeError(f"P0 source fetch failed: {task_id}")
        buggy = init["buggy_source"]
        path = item["q_bank"]["callable"]
        if (task_id, None, 0) not in known:
            save_state(states_path, known, {"instance_id": task_id,
                       "trajectory_id": None, "step": 0, "source": buggy,
                       "body": current_callable_body(buggy, path),
                       "source_sha256": sha256(buggy)})
        elif known[(task_id, None, 0)]["source_sha256"] != init["buggy_source_sha256"]:
            raise RuntimeError(f"buggy source changed: {task_id}")
        for sample in range(16):
            if (task_id, sample, 8) in known:
                continue
            first = frozen[task_id, sample]
            if (task_id, sample, 1) not in known:
                try:
                    source = replace_callable_body(buggy, path, first["completion"])
                    invalid = False
                except InvalidEdit:
                    source, invalid = buggy, True
                if (invalid != (first["status"] == "invalid_body")
                        or sha256(source) != first["final_source_sha256"]):
                    raise RuntimeError(f"cannot reconstruct frozen P1: {task_id}/{sample}")
                save_state(states_path, known, {"instance_id": task_id,
                           "trajectory_id": sample, "step": 1, "source": source,
                           "body": current_callable_body(source, path),
                           "source_sha256": sha256(source),
                           "action_completion": first["completion"],
                           "invalid_body": invalid,
                           "generated_tokens": first["generated_tokens"]})
            for step in range(2, 9):
                if (task_id, sample, step) in known:
                    continue
                prior = known[(task_id, sample, step - 1)]
                seed = int(hashlib.sha256(
                    f"{config['seed']}|{task_id}|{sample}|{step}".encode()).hexdigest()[:8], 16)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                action = generate_action(
                    model, tokenizer, agent_prompt(item, prior["source"], NO_FEEDBACK),
                    device=device, context_tokens=config["context_tokens"],
                    max_new_tokens=config["max_generated_tokens_per_edit"], sample=True,
                    system_prompt=SYSTEM_PROMPT)
                try:
                    source = replace_callable_body(prior["source"], path, action.completion)
                    invalid = False
                except InvalidEdit:
                    source, invalid = prior["source"], True
                save_state(states_path, known, {"instance_id": task_id,
                           "trajectory_id": sample, "step": step, "source": source,
                           "body": current_callable_body(source, path),
                           "source_sha256": sha256(source),
                           "action_completion": action.completion,
                           "invalid_body": invalid,
                           "generated_tokens": len(action.token_ids) - action.prompt_length})
        print(f"{task_id}: 16 capture-only trajectories through P8", flush=True)
    if len(known) != 3612:
        raise RuntimeError(f"incomplete capture: {len(known)} states")
    metadata.update(status="complete", states=len(known),
                    finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("capture complete: 3612 states; no grading run", flush=True)


if __name__ == "__main__":
    main()
