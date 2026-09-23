#!/usr/bin/env python3
"""Fresh P0..P8 closed-loop trajectories: public feedback online, q offline.

This replaces the stopped open-loop capture at commit 885275c. Do not resume
that old run here: its no-feedback states are a different protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import torch

from continue_swesmith_body_trajectories import (
    feedback_from_public, public_fields, save_state, sha256, state_key,
)
from swesmith_agent_edit import InvalidEdit, current_callable_body, replace_callable_body
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import generate_action
from train_swesmith_agent_grpo import agent_prompt, load_policy, load_selected


SYSTEM_PROMPT = (
    "You repair Python functions. Output only the replacement function body, "
    "without def or explanation."
)


def pilot_ids(selection: dict, config: dict) -> list[str]:
    if selection.get("status") != "q_first_frozen" or selection.get("count") != 28:
        raise ValueError("expected frozen 28-task q-valid selection")
    if len(selection.get("ids", [])) != 28 or len(set(selection["ids"])) != 28:
        raise ValueError("selection IDs are incomplete or duplicated")
    n = config["pilot_tasks"]
    if not isinstance(n, int) or not 1 <= n <= 28:
        raise ValueError("pilot_tasks must be between 1 and 28")
    if config["trajectories_per_task"] != 16 or config["steps"] != 8:
        raise ValueError("this protocol requires 16 trajectories and eight edits")
    return selection["ids"][:n]


def public_prompt_item(item: dict) -> dict:
    """Expose only target identity, never q's reference bank or observations."""
    return {"task": item["task"], "q_bank": {
        "module": item["q_bank"]["module"],
        "callable": item["q_bank"]["callable"],
    }}


def validate_feedback_chain(states: dict) -> None:
    """Fail closed if any saved edit lacks its actual preceding observation."""
    for (task_id, sample, step), row in states.items():
        if step == 0:
            if sample is not None or row.get("feedback_given") is not None:
                raise RuntimeError(f"invalid P0 feedback: {task_id}")
            continue
        prior_key = ((task_id, None, 0) if step == 1
                     else (task_id, sample, step - 1))
        prior = states.get(prior_key)
        if prior is None or row.get("feedback_given") != feedback_from_public(prior):
            raise RuntimeError(f"broken public-feedback chain: {task_id}/{sample}/{step}")
        if row.get("invalid_body") and row["source_sha256"] != prior["source_sha256"]:
            raise RuntimeError(f"invalid edit changed source: {task_id}/{sample}/{step}")


def sample_step(*, item: dict, prior: dict, path: list[str],
                model, tokenizer, session,
                config: dict, device: torch.device, seed: int,
                sample: bool = True) -> tuple[dict, object]:
    """One policy edit and public observation, returning the sampled action."""
    feedback = feedback_from_public(prior)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    action = generate_action(
        model, tokenizer, agent_prompt(public_prompt_item(item), prior["source"], feedback),
        device=device, context_tokens=config["context_tokens"],
        max_new_tokens=config["max_generated_tokens_per_edit"], sample=sample,
        system_prompt=SYSTEM_PROMPT,
    )
    try:
        source = replace_callable_body(prior["source"], path, action.completion)
        invalid = False
    except InvalidEdit:
        source, invalid = prior["source"], True
    scored = public_fields(session.call(mode="score", source=source,
                                        include_proxy=False))
    state = {
        "source": source,
        "body": current_callable_body(source, path),
        "source_sha256": sha256(source),
        "action_completion": action.completion,
        "invalid_body": invalid,
        "generated_tokens": len(action.token_ids) - action.prompt_length,
        "feedback_given": feedback,
        **scored,
    }
    return state, action


def run_step(*, item: dict, prior: dict, path: list[str],
             model, tokenizer, session,
             config: dict, device: torch.device, seed: int) -> dict:
    """Capture-only wrapper: save state, not model-token tensors."""
    state, _ = sample_step(item=item, prior=prior, path=path, model=model,
                           tokenizer=tokenizer, session=session, config=config,
                           device=device, seed=seed)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    task_ids = pilot_ids(selection, config)
    if config["max_generated_tokens_per_edit"] != 1024:
        raise ValueError("unexpected body-action generation cap")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    expected = {(task_id, None, 0) for task_id in task_ids}
    expected |= {(task_id, sample, step) for task_id in task_ids
                 for sample in range(16) for step in range(1, 9)}
    metadata = {
        "kind": "fresh_eight_step_closed_loop_public_feedback_q_offline",
        "status": "running", "task_ids": task_ids,
        "tasks": len(task_ids), "trajectories_per_task": 16, "steps": 8,
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_revision": config["model_revision"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "run.json"
    states_path = args.output_dir / "states.jsonl"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        keys = ("kind", "task_ids", "tasks", "trajectories_per_task", "steps",
                "selection_sha256", "config_sha256", "model_revision")
        if any(old.get(key) != metadata[key] for key in keys):
            raise RuntimeError("cannot resume a different trajectory protocol or input")
        if old["status"] == "complete":
            return
        metadata["started_utc"] = old["started_utc"]
    elif states_path.exists():
        raise RuntimeError("states exist without a protocol receipt")
    else:
        receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    known = {}
    if states_path.exists():
        for line in states_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = state_key(row)
            if (key not in expected or key in known or
                    sha256(row["source"]) != row["source_sha256"] or
                    "q" in row or "q_invalid_candidate" in row or
                    not all(k in row for k in ("public", "p_T", "solved"))):
                raise RuntimeError(f"corrupt or non-public saved state: {key}")
            known[key] = row
    validate_feedback_chain(known)

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
        with executor.task_session(task) as session:
            initialized = session.initialize(mode="init")
            if not initialized["gold_self_consistent"]:
                raise RuntimeError(f"gold self-consistency failed: {task_id}")
            buggy = initialized["buggy_source"]
            buggy_hash = initialized["buggy_source_sha256"]
            path = item["q_bank"]["callable"]
            p0_key = (task_id, None, 0)
            if p0_key not in known:
                baseline = public_fields(session.call(mode="score", source=buggy,
                                                      include_proxy=False))
                save_state(states_path, known, {
                    "instance_id": task_id, "trajectory_id": None, "step": 0,
                    "source": buggy, "body": current_callable_body(buggy, path),
                    "source_sha256": sha256(buggy), "invalid_body": False,
                    "feedback_given": None, **baseline,
                })
            elif known[p0_key]["source_sha256"] != buggy_hash:
                raise RuntimeError(f"P0 source changed: {task_id}")
            for sample in range(16):
                for step in range(1, 9):
                    key = (task_id, sample, step)
                    if key in known:
                        continue
                    prior = known[p0_key if step == 1 else (task_id, sample, step - 1)]
                    seed = int(hashlib.sha256(
                        f"{config['seed']}|{task_id}|{sample}|{step}".encode()
                    ).hexdigest()[:8], 16)
                    next_state = run_step(
                        item=item, prior=prior, path=path, model=model,
                        tokenizer=tokenizer, session=session, config=config,
                        device=device, seed=seed,
                    )
                    save_state(states_path, known, {
                        "instance_id": task_id, "trajectory_id": sample,
                        "step": step, **next_state,
                    })
        print(f"{task_id}: 16 closed-loop trajectories through P8", flush=True)
    if known.keys() != expected:
        raise RuntimeError(f"incomplete capture: {len(known)}/{len(expected)} states")
    validate_feedback_chain(known)
    metadata.update(status="complete", states=len(known),
                    states_sha256=hashlib.sha256(states_path.read_bytes()).hexdigest(),
                    finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {len(known)} public-scored states; q remains offline", flush=True)


if __name__ == "__main__":
    main()
