#!/usr/bin/env python3
"""One-edit body-action GRPO over real SWE-smith repository repair tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from grpo_objective import group_advantages
from oracle_proxy_reward import ARMS, terminal_reward
from swesmith_agent_edit import InvalidEdit, replace_callable_body, _node
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import Action, cache_old_and_reference, generate_action, update_policy


@dataclass
class Episode:
    actions: list[Action]
    score: dict
    invalid_edits: int
    final_source_sha256: str


def append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def agent_prompt(item: dict, source: str, feedback: str) -> str:
    q = item["q_bank"]
    node = _node(source, q["callable"])
    lines = source.splitlines()
    first_body_line = node.body[0].lineno
    signature = "\n".join(lines[node.lineno - 1:first_body_line - 1])
    if not signature:
        signature = lines[node.lineno - 1][:node.body[0].col_offset].rstrip()
    current_body = "\n".join(lines[first_body_line - 1:node.end_lineno])
    return (
        "Repair only the body of this Python function. The runtime preserves its "
        "name, signature and decorators. Output only replacement body statements; "
        "do not output def, Markdown, explanation, or a diff.\n"
        f"Issue: {item['task']['problem_statement']}\n"
        f"Target: {q['module']}.{'.'.join(q['callable'])}\n"
        f"Frozen signature:\n{signature}\n"
        f"Current body:\n```python\n{current_body}\n```\n"
        f"Previous tool feedback: {feedback}"
    )


def initialize_item(item: dict, executor) -> None:
    if "buggy_source" in item:
        return
    initialized = executor.call(item["task"], mode="init")
    if not initialized["gold_self_consistent"]:
        raise RuntimeError(f"gold self-consistency failed: {item['task']['instance_id']}")
    item["buggy_source"] = initialized["buggy_source"]
    item["buggy_source_sha256"] = initialized["buggy_source_sha256"]
    if len(agent_prompt(item, item["buggy_source"], "No edits yet.")) > 28_000:
        raise RuntimeError(f"prompt text too long: {item['task']['instance_id']}")


def episode(item: dict, model, tokenizer, executor, *, config: dict,
            device: torch.device, sample: bool, proxy: bool) -> Episode:
    initialize_item(item, executor)
    source = item["buggy_source"]
    action = generate_action(
        model, tokenizer, agent_prompt(item, source, "No edits yet."),
        device=device, context_tokens=config["context_tokens"],
        max_new_tokens=config["max_generated_tokens_per_edit"], sample=sample,
        system_prompt="You repair Python functions. Output only the replacement function body, without def or explanation.",
    )
    try:
        source = replace_callable_body(source, item["q_bank"]["callable"], action.completion)
    except InvalidEdit:
        failed = {"status": "invalid_body", "p_T": 0.0, "solved": False}
        if proxy:
            failed["q"] = 0.0
        return Episode([action], failed, 1, hashlib.sha256(source.encode()).hexdigest())
    final = executor.call(item["task"], mode="score", source=source,
                          buggy_source_sha256=item["buggy_source_sha256"],
                          q_bank=item["q_bank"] if proxy else None, include_proxy=proxy)
    return Episode([action], final, 0, hashlib.sha256(source.encode()).hexdigest())


def load_selected(tasks_path: Path, q_path: Path, ids: list[str]) -> list[dict]:
    tasks = {row["instance_id"]: row for line in tasks_path.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line))}
    q = {row["instance_id"]: row for line in q_path.read_text(encoding="utf-8").splitlines()
         if (row := json.loads(line)) and row["status"] == "q_valid"}
    return [{"task": tasks[task_id], "q_bank": q[task_id]} for task_id in ids]


def load_policy(config: dict, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = snapshot_download(config["model"], revision=config["model_revision"],
                                 local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.bfloat16, trust_remote_code=False,
    ).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = get_peft_model(base, LoraConfig(
        r=config["adapter"]["rank"], lora_alpha=2 * config["adapter"]["rank"],
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
    ))
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--q-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["max_edits"] != 1:
        raise ValueError("body-action survival requires exactly one edit")
    if args.smoke_updates and (args.smoke_updates != 1 or config["group_size"] != 4
                               or config["tasks_per_update"] != 1):
        raise ValueError("body-action smoke requires one task, four rollouts, one update")
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    if selected["status"] != "q_first_frozen" or selected["count"] != len(selected["ids"]):
        raise ValueError("selection not frozen")
    if not args.smoke_updates and (selected["count"] != 28 or selected["train_count"] != 20
                                   or selected["heldout_count"] != 8):
        raise ValueError("survival experiment requires frozen 20+8 tasks")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    executor = SWESmithModalExecutor()
    items = load_selected(args.tasks, args.q_results, selected["ids"])
    tokenizer, model = load_policy(config, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["learning_rate"])
    args.output_dir.mkdir(parents=True)
    append(args.output_dir / "metrics.jsonl", {"event": "setup", "arm": args.arm,
           "formal": not bool(args.smoke_updates), "count": len(items),
           "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
           "model_revision": config["model_revision"], "time_utc": datetime.now(timezone.utc).isoformat()})
    train = [item for item in items if selected["splits"][item["task"]["instance_id"]] == "train"]
    heldout = [item for item in items if selected["splits"][item["task"]["instance_id"]] == "heldout"]
    updates = args.smoke_updates or config["updates_per_arm"]
    for update in range(updates):
        started = time.perf_counter()
        chosen = [train[(update * config["tasks_per_update"] + index) % len(train)]
                  for index in range(config["tasks_per_update"])]
        episodes = [episode(item, model, tokenizer, executor, config=config, device=device,
                            sample=True, proxy=args.arm == "semantic")
                    for item in chosen for _ in range(config["group_size"])]
        rewards = torch.tensor([terminal_reward(args.arm, solved=ep.score["solved"],
                                               public_pass_fraction=ep.score["p_T"],
                                               reference_agreement=ep.score.get("q"))
                                for ep in episodes], device=device)
        advantages = group_advantages(rewards, group_size=config["group_size"])
        actions = [action for ep in episodes for action in ep.actions]
        action_advantages = [float(a) for a in advantages.tolist()
                             for _ in range(config["max_edits"])]
        for ep in episodes:
            append(args.output_dir / "rollouts.jsonl", {"update": update + 1,
                   "score": ep.score, "invalid_edits": ep.invalid_edits,
                   "final_source_sha256": ep.final_source_sha256,
                   "completions": [a.completion for a in ep.actions]})
        valid_count = sum(ep.invalid_edits == 0 for ep in episodes)
        distinct_rewards = len(set(rewards.tolist()))
        if args.smoke_updates and (valid_count < 2 or distinct_rewards < 2):
            append(args.output_dir / "metrics.jsonl", {"event": "smoke_gate", "passed": False,
                   "valid_bodies": valid_count, "distinct_rewards": distinct_rewards,
                   "reason": "insufficient_valid_bodies_or_reward_variance"})
            raise RuntimeError("body-action smoke failed before optimizer update")
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable] if args.smoke_updates else []
        cache_old_and_reference(model, actions, device)
        metrics = update_policy(model, optimizer, actions, action_advantages,
                                device=device, config=config)
        if args.smoke_updates:
            max_delta = max(float((p.detach() - old).abs().max())
                            for p, old in zip(trainable, before))
            metrics["adapter_max_abs_delta"] = max_delta
        append(args.output_dir / "metrics.jsonl", {"event": "train", "update": update + 1,
               "rewards": rewards.tolist(), "solved": sum(ep.score["solved"] for ep in episodes),
               "invalid_edits": sum(ep.invalid_edits for ep in episodes),
               "valid_bodies": valid_count, "distinct_rewards": distinct_rewards,
               "seconds": time.perf_counter() - started, **metrics})
        if args.smoke_updates:
            passed = metrics["grad_norm"] > 0 and metrics["adapter_max_abs_delta"] > 0
            append(args.output_dir / "metrics.jsonl", {"event": "smoke_gate", "passed": passed,
                   "valid_bodies": valid_count, "distinct_rewards": distinct_rewards,
                   "grad_norm": metrics["grad_norm"],
                   "adapter_max_abs_delta": metrics["adapter_max_abs_delta"]})
            if not passed:
                raise RuntimeError("body-action smoke failed to change LoRA parameters")
        if (update + 1) % config["eval_every_updates"] == 0 or update + 1 == updates:
            successes = 0
            for item in heldout:
                scored = episode(item, model, tokenizer, executor, config=config,
                                 device=device, sample=False, proxy=False)
                successes += scored.score["solved"]
            append(args.output_dir / "metrics.jsonl", {"event": "eval", "update": update + 1,
                   "solved": successes, "total": len(heldout),
                   "pass_at_1": successes / len(heldout) if heldout else None})
            model.save_pretrained(args.output_dir / f"adapter_update_{update+1:03d}")
        print(f"{args.arm} update {update+1}/{updates} complete", flush=True)


if __name__ == "__main__":
    main()
