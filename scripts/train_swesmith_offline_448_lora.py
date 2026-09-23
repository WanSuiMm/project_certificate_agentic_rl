#!/usr/bin/env python3
"""Train Test and Semantic LoRA arms from frozen P1 actions without new scoring.

This is fixed-data offline policy optimization, not on-policy GRPO. The census
saved decoded completions rather than sampled token IDs, so they are retokenized.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random

import torch

from grpo_objective import group_advantages
from oracle_proxy_reward import terminal_reward
from swesmith_agent_edit import InvalidEdit, replace_callable_body
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import Action, append_jsonl, completion_logp, update_policy
from train_swesmith_agent_grpo import agent_prompt, load_policy, load_selected


SYSTEM_PROMPT = "You repair Python functions. Output only the replacement function body, without def or explanation."


def frozen_rows(census: Path, selection: dict) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in (census / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 448:
        raise RuntimeError("expected exactly 448 frozen P1 rows")
    grouped = {task_id: [] for task_id in selection["ids"]}
    for row in rows:
        if row["instance_id"] not in grouped or row["status"] not in {
                "scored", "invalid_candidate", "invalid_body"}:
            raise RuntimeError("census contains an unselected task or scorer error")
        for arm in ("test", "semantic"):
            expected = terminal_reward(
                arm, solved=row["score"]["solved"],
                public_pass_fraction=row["score"]["p_T"],
                reference_agreement=row["score"]["q"])
            if abs(row[f"{arm}_reward"] - expected) > 1e-8:
                raise RuntimeError("stored reward does not match frozen formula")
        grouped[row["instance_id"]].append(row)
    for task_id, block in grouped.items():
        if len(block) != 16 or {row["sample"] for row in block} != set(range(16)):
            raise RuntimeError(f"incomplete frozen P1 group: {task_id}")
        block.sort(key=lambda row: row["sample"])
    return grouped


def p0_sources(path: Path, items: list[dict]) -> dict[str, str]:
    sources = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (row["instance_id"] in sources or
                    hashlib.sha256(row["source"].encode()).hexdigest() != row["sha256"]):
                raise RuntimeError("duplicate or corrupt cached P0 source")
            sources[row["instance_id"]] = row["source"]
    if len(sources) == len(items):
        return sources
    executor = SWESmithModalExecutor()
    for item in items:
        task_id = item["task"]["instance_id"]
        if task_id in sources:
            continue
        # Data retrieval only: no public tests, q observation, or rollout.
        result = executor.call(item["task"], mode="source_only")
        if result["status"] != "source_only":
            raise RuntimeError(f"could not load P0 source: {task_id}")
        source = result["buggy_source"]
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != result["buggy_source_sha256"]:
            raise RuntimeError(f"P0 hash mismatch: {task_id}")
        append_jsonl(path, {"instance_id": task_id, "source": source, "sha256": digest})
        sources[task_id] = source
        print(f"P0 source loaded {len(sources)}/{len(items)}", flush=True)
    return sources


def retokenized_action(tokenizer, prompt: str, completion: str, context_tokens: int) -> Action:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    suffix = tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")
    if suffix.numel() == 0 or prefix.numel() + suffix.numel() > context_tokens:
        raise RuntimeError("empty or over-context retokenized completion")
    ids = torch.cat((prefix[0], suffix[0])).cpu()
    return Action(ids, int(prefix.shape[1]), completion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "census", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (config["protocol"] != "swesmith_offline_448_lora_v01" or
            config["group_size"] != 16 or config["updates_per_arm"] != 20 or
            config["ppo_epochs"] != 1):
        raise ValueError("wrong offline update protocol")
    if (selection["status"] != "q_first_frozen" or selection["count"] != 28 or
            selection["train_count"] != 20 or selection["heldout_count"] != 8):
        raise ValueError("wrong frozen 20/8 split")
    if json.loads((args.census / "run.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError("448-candidate census is incomplete")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    grouped = frozen_rows(args.census, selection)
    items = load_selected(args.tasks, args.q_results, selection["ids"])
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    append_jsonl(metrics_path, {
        "event": "setup", "time_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"], "census_sha256": hashlib.sha256(
            (args.census / "results.jsonl").read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "train_candidates": 320, "heldout_candidates": 128,
        "new_rollouts": 0, "new_score_calls": 0,
        "claim": "offline fixed-data LoRA update, not on-policy GRPO"})
    sources = p0_sources(args.output_dir / "p0_sources.jsonl", items)
    by_id = {item["task"]["instance_id"]: item for item in items}
    for task_id in selection["ids"]:
        source = sources[task_id]
        path = by_id[task_id]["q_bank"]["callable"]
        for row in grouped[task_id]:
            try:
                p1 = replace_callable_body(source, path, row["completion"])
                invalid = False
            except InvalidEdit:
                p1, invalid = source, True
            if (invalid != (row["status"] == "invalid_body") or
                    hashlib.sha256(p1.encode()).hexdigest() != row["final_source_sha256"]):
                raise RuntimeError(f"frozen P1 reconstruction failed: {task_id}/{row['sample']}")
    append_jsonl(metrics_path, {"event": "p1_verified", "candidates": 448})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer, model = load_policy(config, device)
    train_ids = [x for x in selection["ids"] if selection["splits"][x] == "train"]
    random.Random(config["seed"]).shuffle(train_ids)
    actions = {}
    for task_id in train_ids:
        prompt = agent_prompt(by_id[task_id], sources[task_id], "No edits yet.")
        actions[task_id] = [retokenized_action(
            tokenizer, prompt, row["completion"], config["context_tokens"])
            for row in grouped[task_id]]
    append_jsonl(metrics_path, {"event": "retokenized", "actions": 320,
                "caveat": "original sampled token IDs were not saved"})
    # Cache the frozen base-policy log probabilities before either arm updates.
    model.eval()
    with torch.inference_mode():
        for index, task_id in enumerate(train_ids, 1):
            for action in actions[task_id]:
                action.old_logp = completion_logp(model, action, device).detach()
                with model.disable_adapter():
                    action.ref_logp = completion_logp(model, action, device).detach()
            print(f"base logprobs {index}/20 groups", flush=True)
    append_jsonl(metrics_path, {"event": "base_logprobs_cached", "actions": 320})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for arm in ("test", "semantic"):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        _, model = load_policy(config, device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"])
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        for update, task_id in enumerate(train_ids, 1):
            rewards = torch.tensor(
                [row[f"{arm}_reward"] for row in grouped[task_id]], device=device)
            advantages = group_advantages(rewards, group_size=16).tolist()
            metrics = update_policy(
                model, optimizer, actions[task_id], advantages,
                device=device, config=config)
            append_jsonl(metrics_path, {
                "event": "train", "arm": arm, "update": update,
                "task_id": task_id, "distinct_rewards": len(set(rewards.tolist())),
                "nonzero_advantages": sum(abs(x) > 0 for x in advantages),
                **metrics})
            print(f"{arm} update {update}/20; grad={metrics['grad_norm']:.6g}", flush=True)
            if update in (1, 10, 20):
                model.save_pretrained(arm_dir / f"adapter_update_{update:03d}")
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    append_jsonl(metrics_path, {
        "event": "complete", "time_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
