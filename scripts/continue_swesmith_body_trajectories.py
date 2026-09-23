#!/usr/bin/env python3
"""Continue frozen one-edit candidates to eight edits using public feedback only.

No reference bank or q value is passed to generation or online scoring.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import torch

from swesmith_agent_edit import InvalidEdit, current_callable_body, replace_callable_body
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import generate_action
from train_swesmith_agent_grpo import agent_prompt, append, load_policy, load_selected


SYSTEM_PROMPT = "You repair Python functions. Output only the replacement function body, without def or explanation."


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def feedback_from_public(score: dict) -> str:
    public = score["public"]
    if public.get("valid"):
        return (f"Public tests: {public['passed']}/{public['total']} passed; "
                f"exitcode={public['exitcode']}.")
    return f"Public tests unavailable: {public.get('reason', 'invalid_observation')}."


def public_fields(result: dict) -> dict:
    if "q" in result or "q_invalid_candidate" in result:
        raise ValueError("online rollout must not consume q")
    return {"public": result["public"], "p_T": result["p_T"],
            "solved": result["solved"], "public_status": result["status"]}


def state_key(row: dict) -> tuple[str, int | None, int]:
    return row["instance_id"], row["trajectory_id"], row["step"]


def save_state(path: Path, known: dict, row: dict) -> None:
    key = state_key(row)
    if key in known:
        raise RuntimeError(f"duplicate state: {key}")
    if sha256(row["source"]) != row["source_sha256"]:
        raise RuntimeError(f"source hash mismatch before save: {key}")
    append(path, row)
    known[key] = row


def load_census(path: Path, selection: dict) -> dict[tuple[str, int], dict]:
    status = json.loads((path / "run.json").read_text(encoding="utf-8"))["status"]
    if status != "complete":
        raise RuntimeError(f"one-edit census is not complete: {status}")
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(task_id, sample) for task_id in selection["ids"] for sample in range(16)}
    found = {(row["instance_id"], row["sample"]) for row in rows}
    if len(rows) != 448 or found != expected:
        raise RuntimeError("one-edit census does not contain exactly 28 x 16 unique candidates")
    if any(row["status"] not in {"scored", "invalid_candidate", "invalid_body"} for row in rows):
        raise RuntimeError("one-edit census contains infrastructure or task errors")
    return {(row["instance_id"], row["sample"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--q-results", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["status"] != "q_first_frozen" or selection["count"] != 28:
        raise ValueError("wrong frozen selection")
    if config["max_edits"] != 1 or config["max_generated_tokens_per_edit"] != 1024:
        raise ValueError("wrong body-action configuration")
    census = load_census(args.census, selection)
    census_sha = hashlib.sha256((args.census / "results.jsonl").read_bytes()).hexdigest()
    metadata = {"kind": "eight_step_observational_trajectory_public_only",
                "status": "running", "tasks": 28, "trajectories_per_task": 16, "steps": 8,
                "source_census_sha256": census_sha,
                "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "model_revision": config["model_revision"],
                "started_utc": datetime.now(timezone.utc).isoformat()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "run.json"
    states_path = args.output_dir / "states.jsonl"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if any(old[key] != metadata[key] for key in ("kind", "tasks", "trajectories_per_task",
                                                  "steps", "source_census_sha256",
                                                  "selection_sha256", "config_sha256", "model_revision")):
            raise RuntimeError("cannot resume with changed inputs")
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
            known[key] = row

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer, model = load_policy(config, device)
    executor = SWESmithModalExecutor()
    items = load_selected(args.tasks, args.q_results, selection["ids"])
    for item in items:
        task = item["task"]
        task_id = task["instance_id"]
        if all((task_id, sample, 8) in known for sample in range(16)):
            continue
        init = executor.call(task, mode="init")
        if not init["gold_self_consistent"]:
            raise RuntimeError(f"gold self-consistency failed: {task_id}")
        buggy = init["buggy_source"]
        path = item["q_bank"]["callable"]
        if (task_id, None, 0) not in known:
            baseline = public_fields(executor.call(task, mode="score", source=buggy,
                                                    buggy_source_sha256=init["buggy_source_sha256"],
                                                    include_proxy=False))
            save_state(states_path, known, {"instance_id": task_id, "trajectory_id": None,
                       "step": 0, "source": buggy, "body": current_callable_body(buggy, path),
                       "source_sha256": sha256(buggy), **baseline})
        elif known[(task_id, None, 0)]["source_sha256"] != init["buggy_source_sha256"]:
            raise RuntimeError(f"buggy source changed: {task_id}")
        for sample in range(16):
            if (task_id, sample, 8) in known:
                continue
            first = census[(task_id, sample)]
            if (task_id, sample, 1) not in known:
                try:
                    source = replace_callable_body(buggy, path, first["completion"])
                    invalid = False
                except InvalidEdit:
                    source, invalid = buggy, True
                if invalid != (first["status"] == "invalid_body") or sha256(source) != first["final_source_sha256"]:
                    raise RuntimeError(f"cannot reconstruct frozen P1: {task_id}/{sample}")
                if invalid:
                    first_public = public_fields(executor.call(task, mode="score", source=source,
                                                                 buggy_source_sha256=init["buggy_source_sha256"],
                                                                 include_proxy=False))
                else:
                    # The census already ran the public tests. Discard its q field.
                    first_public = public_fields({key: first["score"][key]
                                                  for key in ("public", "p_T", "solved", "status")})
                save_state(states_path, known, {"instance_id": task_id, "trajectory_id": sample,
                           "step": 1, "source": source, "body": current_callable_body(source, path),
                           "source_sha256": sha256(source), "action_completion": first["completion"],
                           "invalid_body": invalid, "generated_tokens": first["generated_tokens"],
                           "feedback_given": "No edits yet.", **first_public})
            for step in range(2, 9):
                prior = known[(task_id, sample, step - 1)]
                if (task_id, sample, step) in known:
                    continue
                feedback = feedback_from_public(prior)
                seed = int(hashlib.sha256(f"{config['seed']}|{task_id}|{sample}|{step}".encode()).hexdigest()[:8], 16)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                action = generate_action(
                    model, tokenizer, agent_prompt(item, prior["source"], feedback),
                    device=device, context_tokens=config["context_tokens"],
                    max_new_tokens=config["max_generated_tokens_per_edit"], sample=True,
                    system_prompt=SYSTEM_PROMPT)
                try:
                    source = replace_callable_body(prior["source"], path, action.completion)
                    invalid = False
                except InvalidEdit:
                    source, invalid = prior["source"], True
                public = public_fields(executor.call(task, mode="score", source=source,
                                                     buggy_source_sha256=init["buggy_source_sha256"],
                                                     include_proxy=False))
                save_state(states_path, known, {"instance_id": task_id, "trajectory_id": sample,
                           "step": step, "source": source, "body": current_callable_body(source, path),
                           "source_sha256": sha256(source), "action_completion": action.completion,
                           "invalid_body": invalid, "generated_tokens": len(action.token_ids) - action.prompt_length,
                           "feedback_given": feedback, **public})
        print(f"{task_id}: 16 trajectories through P8", flush=True)
    if len(known) != 28 + 28 * 16 * 8:
        raise RuntimeError(f"incomplete trajectory state count: {len(known)}")
    metadata.update(status="complete", states=len(known),
                    finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {len(known)} states", flush=True)


if __name__ == "__main__":
    main()
