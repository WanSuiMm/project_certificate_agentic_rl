#!/usr/bin/env python3
"""Online GRPO over real SWE-smith repository repair episodes.

Each sampled episode has three sequential model edits. Public test feedback is
visible between edits; frozen reference q is used only for terminal reward.
"""

from __future__ import annotations

import argparse
import ast
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
from swesmith_agent_edit import InvalidEdit, replace_callable, _node
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


def code_excerpt(source: str, path: list[str]) -> str:
    node = _node(source, path)
    lines = source.splitlines()
    first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    return "\n".join(lines[first - 1:node.end_lineno])


def agent_prompt(item: dict, source: str, feedback: str) -> str:
    q = item["q_bank"]
    snippet = code_excerpt(source, q["callable"])
    return (
        "Repair this actual SWE-smith Python repository task. Output exactly one complete "
        "replacement def block for the named function or method. Preserve its signature. "
        "Do not output explanations, tests, or a diff.\n"
        f"Issue: {item['task']['problem_statement']}\n"
        f"Target: {q['module']}.{'.'.join(q['callable'])}\n"
        f"Current function:\n```python\n{snippet}\n```\n"
        f"Previous tool feedback: {feedback}"
    )


def episode(item: dict, model, tokenizer, executor, *, config: dict,
            device: torch.device, sample: bool, proxy: bool) -> Episode:
    source = item["buggy_source"]
    feedback = "No edits yet."
    actions = []
    invalid = 0
    for _ in range(config["max_edits"]):
        action = generate_action(model, tokenizer, agent_prompt(item, source, feedback),
                                 device=device, context_tokens=config["context_tokens"],
                                 max_new_tokens=config["max_generated_tokens_per_edit"], sample=sample)
        actions.append(action)
        try:
            proposal = replace_callable(source, item["q_bank"]["callable"], action.completion)
        except (InvalidEdit, SyntaxError) as exc:
            invalid += 1
            feedback = f"Edit rejected: {str(exc)[:200]}. No code changed."
            continue
        source = proposal
        public = executor.call(item["task"], mode="score", source=source,
                               buggy_source_sha256=item["buggy_source_sha256"],
                               include_proxy=False)
        feedback = (f"Official public tests: {public['public']['passed']}/"
                    f"{public['public']['total']} passed; exitcode={public['public']['exitcode']}.")
    final = executor.call(item["task"], mode="score", source=source,
                          buggy_source_sha256=item["buggy_source_sha256"],
                          q_bank=item["q_bank"] if proxy else None, include_proxy=proxy)
    return Episode(actions, final, invalid, hashlib.sha256(source.encode()).hexdigest())


def load_selected(tasks_path: Path, q_path: Path, ids: list[str]) -> list[dict]:
    tasks = {row["instance_id"]: row for line in tasks_path.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line))}
    q = {row["instance_id"]: row for line in q_path.read_text(encoding="utf-8").splitlines()
         if (row := json.loads(line)) and row["status"] == "q_valid"}
    return [{"task": tasks[task_id], "q_bank": q[task_id]} for task_id in ids]


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
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    if selected["status"] != "q_first_frozen" or selected["count"] != len(selected["ids"]):
        raise ValueError("selection not frozen")
    if not args.smoke_updates and selected["count"] not in (40, 52, 64):
        raise ValueError("formal experiment requires 32+8, 40+12, or 48+16 tasks")
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
    for item in items:
        initialized = executor.call(item["task"], mode="init")
        if not initialized["gold_self_consistent"]:
            raise RuntimeError(f"gold self-consistency failed: {item['task']['instance_id']}")
        item["buggy_source"] = initialized["buggy_source"]
        item["buggy_source_sha256"] = initialized["buggy_source_sha256"]
        if len(agent_prompt(item, item["buggy_source"], "No edits yet.")) > 28_000:
            raise RuntimeError(f"prompt text too long: {item['task']['instance_id']}")
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
                            sample=True, proxy=True)
                    for item in chosen for _ in range(config["group_size"])]
        rewards = torch.tensor([terminal_reward(args.arm, solved=ep.score["solved"],
                                               public_pass_fraction=ep.score["p_T"],
                                               reference_agreement=ep.score["q"])
                                for ep in episodes], device=device)
        advantages = group_advantages(rewards, group_size=config["group_size"])
        actions = [action for ep in episodes for action in ep.actions]
        action_advantages = [float(a) for a in advantages.tolist()
                             for _ in range(config["max_edits"])]
        cache_old_and_reference(model, actions, device)
        metrics = update_policy(model, optimizer, actions, action_advantages,
                                device=device, config=config)
        append(args.output_dir / "metrics.jsonl", {"event": "train", "update": update + 1,
               "rewards": rewards.tolist(), "solved": sum(ep.score["solved"] for ep in episodes),
               "invalid_edits": sum(ep.invalid_edits for ep in episodes),
               "seconds": time.perf_counter() - started, **metrics})
        for ep in episodes:
            append(args.output_dir / "rollouts.jsonl", {"update": update + 1,
                   "score": ep.score, "final_source_sha256": ep.final_source_sha256,
                   "completions": [a.completion for a in ep.actions]})
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
