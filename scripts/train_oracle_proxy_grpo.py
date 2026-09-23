#!/usr/bin/env python3
"""Three-arm, three-edit Function-SWE GRPO survival training.

Requires a frozen 48/16 task manifest and a separately attested isolated
worker command. This script is not a task curator or a sandbox implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import torch

from function_swe import TaskError, public_prompt, replace_top_level_function, sha256_text
from function_swe_executor import CommandExecutor
from function_swe_manifest import file_sha256, load_manifest, task_schedule
from grpo_objective import clipped_grpo_loss, group_advantages, token_log_probs
from oracle_proxy_reward import ARMS, terminal_reward


@dataclass
class Action:
    token_ids: torch.Tensor  # 1D, stored on CPU
    prompt_length: int
    completion: str
    old_logp: torch.Tensor | None = None
    ref_logp: torch.Tensor | None = None


@dataclass
class Episode:
    task_id: str
    actions: list[Action]
    final_score: dict[str, Any]
    final_source_sha256: str
    invalid_edits: int


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_action(model: Any, tokenizer: Any, prompt: str, *, device: torch.device,
                    context_tokens: int, max_new_tokens: int, sample: bool) -> Action:
    messages = [
        {"role": "system", "content": "You are a Python repair agent. Output exactly one complete replacement function definition."},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt")
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    prompt_length = encoded.shape[1]
    if prompt_length + max_new_tokens > context_tokens:
        raise TaskError(f"prompt exceeds context budget: {prompt_length}+{max_new_tokens}>{context_tokens}")
    ids = encoded.to(device)
    model.eval()
    sampling_options = {"temperature": 1.0, "top_p": 1.0} if sample else {}
    with torch.inference_mode():
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **sampling_options,
        )
    completion_ids = generated[0, prompt_length:]
    if completion_ids.numel() == 0:
        raise RuntimeError("model produced an empty completion")
    completion = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
    return Action(generated[0].detach().cpu().clone(), prompt_length, completion)


def rollout(task: dict[str, Any], model: Any, tokenizer: Any, executor: CommandExecutor,
            *, device: torch.device, config: dict[str, Any], sample: bool,
            include_proxy: bool = True) -> Episode:
    source = task["buggy_source"]
    feedback = "No edits yet."
    actions: list[Action] = []
    invalid_edits = 0
    for _ in range(config["max_edits"]):
        action = generate_action(
            model, tokenizer, public_prompt(task, source, feedback), device=device,
            context_tokens=config["context_tokens"],
            max_new_tokens=config["max_generated_tokens_per_edit"], sample=sample,
        )
        actions.append(action)
        try:
            source = replace_top_level_function(source, task["target_function"], action.completion)
        except TaskError as exc:
            invalid_edits += 1
            feedback = f"Invalid edit: {exc}. The source did not change."
            continue
        public = executor.score(task, source, terminal=False)
        feedback = f"Public tests: {public['public_passed']}/{public['public_total']} passed."
    final_score = executor.score(task, source, terminal=True, include_proxy=include_proxy)
    return Episode(task["task_id"], actions, final_score, sha256_text(source), invalid_edits)


def completion_logp(model: Any, action: Action, device: torch.device) -> torch.Tensor:
    ids = action.token_ids.unsqueeze(0).to(device)
    mask = torch.ones_like(ids)
    values = token_log_probs(model, ids, mask)[:, action.prompt_length - 1:]
    if values.shape[1] != ids.shape[1] - action.prompt_length:
        raise RuntimeError("completion/log-probability alignment failed")
    return values


def cache_old_and_reference(model: Any, actions: list[Action], device: torch.device) -> None:
    model.eval()
    for action in actions:
        with torch.inference_mode():
            old = completion_logp(model, action, device)
            with model.disable_adapter():
                reference = completion_logp(model, action, device)
        action.old_logp = old.detach().clone()
        action.ref_logp = reference.detach().clone()


def update_policy(model: Any, optimizer: Any, actions: list[Action], advantages: list[float],
                  *, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    if len(actions) != len(advantages) or not actions:
        raise ValueError("action/advantage mismatch")
    parameters = [p for p in model.parameters() if p.requires_grad]
    final_loss = final_kl = final_clip = final_grad = 0.0
    for _ in range(config["ppo_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = kl_sum = clip_sum = 0.0
        for action, advantage in zip(actions, advantages):
            if action.old_logp is None or action.ref_logp is None:
                raise RuntimeError("old/reference policy log-probabilities missing")
            new = completion_logp(model, action, device)
            mask = torch.ones_like(new)
            loss, metrics = clipped_grpo_loss(
                new, action.old_logp, action.ref_logp, mask,
                torch.tensor([advantage], device=device),
                clip_epsilon=config["clip_epsilon"], kl_beta=config["reference_kl_beta"],
            )
            (loss / len(actions)).backward()
            loss_sum += float(loss.detach())
            kl_sum += float(metrics["mean_kl"])
            clip_sum += float(metrics["clip_fraction"])
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
        optimizer.step()
        final_loss = loss_sum / len(actions)
        final_kl = kl_sum / len(actions)
        final_clip = clip_sum / len(actions)
        final_grad = float(grad_norm)
    return {"loss": final_loss, "mean_kl": final_kl, "clip_fraction": final_clip,
            "grad_norm": final_grad}


def evaluate(model: Any, tokenizer: Any, tasks: list[dict[str, Any]], executor: CommandExecutor,
             *, device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    solved = 0
    details = []
    for task in tasks:
        episode = rollout(task, model, tokenizer, executor, device=device, config=config,
                          sample=False, include_proxy=False)
        y = bool(episode.final_score["solved"])
        solved += y
        details.append({"task_id": task["task_id"], "solved": y,
                        "final_source_sha256": episode.final_source_sha256})
    return {"solved": solved, "total": len(tasks), "pass_at_1": solved / len(tasks),
            "tasks": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--executor-command-json", required=True,
                        help="JSON array of a separately isolated scorer command")
    parser.add_argument("--isolation-receipt", required=True, type=Path,
                        help="JSON receipt from a successful isolation smoke")
    parser.add_argument("--smoke-updates", type=int, default=0,
                        help="Non-formal smoke: run only this many updates")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != "oracle_proxy_grpo_survival_v01":
        raise ValueError("wrong config protocol")
    if any(config[key] != value for key, value in (("train_tasks", 48), ("heldout_tasks", 16),
                                                  ("group_size", 4), ("tasks_per_update", 4),
                                                  ("max_edits", 3))):
        raise ValueError("frozen survival dimensions changed")
    if config["sampling_temperature"] != 1.0 or config["sampling_top_p"] != 1.0:
        raise ValueError("sampling must match the unwarped policy log-probabilities")
    if config["rollouts_per_arm"] != config["updates_per_arm"] * config["tasks_per_update"] * config["group_size"]:
        raise ValueError("rollout budget does not match the update schedule")
    receipt = json.loads(args.isolation_receipt.read_text(encoding="utf-8"))
    if receipt.get("isolated_code_execution") is not True or receipt.get("smoke_passed") is not True:
        raise RuntimeError("no passing isolated executor receipt")
    command = json.loads(args.executor_command_json)
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("executor command must be a JSON string array")
    _, tasks = load_manifest(args.manifest)
    train = {task["task_id"]: task for task in tasks if task["split"] == "train"}
    heldout = [task for task in tasks if task["split"] == "heldout"]
    schedule = task_schedule(list(train), seed=config["seed"],
                             updates=config["updates_per_arm"],
                             tasks_per_update=config["tasks_per_update"])
    total_updates = args.smoke_updates or config["updates_per_arm"]
    if total_updates < 1 or total_updates > config["updates_per_arm"]:
        raise ValueError("invalid smoke update count")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if not args.smoke_updates and free_bytes < 24 * 1024**3:
            raise RuntimeError("formal run requires at least 24 GiB free VRAM")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["model_revision"], torch_dtype=torch.bfloat16,
        trust_remote_code=False,
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
    executor = CommandExecutor(command)
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True,
                                 check=False).stdout.strip()
    metadata = {
        "event": "setup", "arm": args.arm, "formal": not bool(args.smoke_updates),
        "config_sha256": file_sha256(args.config), "manifest_sha256": file_sha256(args.manifest),
        "isolation_receipt_sha256": file_sha256(args.isolation_receipt),
        "code_commit": code_commit, "model": config["model"],
        "model_revision": config["model_revision"], "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "setup.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "task_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    metrics_path = args.output_dir / "metrics.jsonl"
    append_jsonl(metrics_path, metadata)
    for update in range(total_updates + 1):
        if update == 0 or update % config["eval_every_updates"] == 0 or update == total_updates:
            result = evaluate(model, tokenizer, heldout, executor, device=device, config=config)
            append_jsonl(metrics_path, {"event": "eval", "update": update, **result})
        if update == total_updates:
            break
        started = time.perf_counter()
        episodes = []
        for task_id in schedule[update]:
            for _ in range(config["group_size"]):
                episodes.append(rollout(train[task_id], model, tokenizer, executor,
                                        device=device, config=config, sample=True))
        rewards = torch.tensor([
            terminal_reward(args.arm, solved=bool(ep.final_score["solved"]),
                            public_pass_fraction=ep.final_score["p_T"],
                            reference_agreement=ep.final_score["q"])
            for ep in episodes
        ], device=device)
        advantages = group_advantages(rewards, group_size=config["group_size"])
        actions = [action for episode in episodes for action in episode.actions]
        action_advantages = [float(advantage) for advantage in advantages.tolist()
                             for _ in range(config["max_edits"])]
        cache_old_and_reference(model, actions, device)
        train_metrics = update_policy(model, optimizer, actions, action_advantages,
                                      device=device, config=config)
        append_jsonl(metrics_path, {
            "event": "train", "update": update + 1, "arm": args.arm,
            "task_ids": schedule[update], "rollouts": len(episodes),
            "rewards": rewards.tolist(), "solved": sum(bool(e.final_score["solved"]) for e in episodes),
            "informative_groups": sum(
                len(set(rewards[i:i+config["group_size"]].tolist())) > 1
                for i in range(0, len(rewards), config["group_size"])
            ),
            "invalid_edits": sum(e.invalid_edits for e in episodes),
            "seconds": time.perf_counter() - started, **train_metrics,
        })
        for episode in episodes:
            append_jsonl(args.output_dir / "rollouts.jsonl", {
                "update": update + 1, "task_id": episode.task_id,
                "completions": [a.completion for a in episode.actions],
                "final_source_sha256": episode.final_source_sha256,
                "score": episode.final_score,
            })
        if (update + 1) % config["eval_every_updates"] == 0 or update + 1 == total_updates:
            model.save_pretrained(args.output_dir / f"adapter_update_{update+1:03d}")
        print(f"{args.arm} update {update+1}/{total_updates}: "
              f"solved={sum(bool(e.final_score['solved']) for e in episodes)}/16", flush=True)


if __name__ == "__main__":
    main()
