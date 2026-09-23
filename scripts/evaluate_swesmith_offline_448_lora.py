#!/usr/bin/env python3
"""Heldout candidate log-probability ranking; no generation or environment scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from train_oracle_proxy_grpo import completion_logp
from train_swesmith_agent_grpo import agent_prompt, load_policy, load_selected
from train_swesmith_offline_448_lora import frozen_rows, retokenized_action


def pairwise(logps: list[float], values: list[float]) -> tuple[int, int]:
    concordant = discordant = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dv = values[i] - values[j]
            dp = logps[i] - logps[j]
            if dv == 0 or dp == 0:
                continue
            if dv * dp > 0:
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "census",
                 "training-dir", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    grouped = frozen_rows(args.census, selection)
    sources = {}
    for line in (args.training_dir / "p0_sources.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sources[row["instance_id"]] = row["source"]
    items = load_selected(args.tasks, args.q_results, selection["ids"])
    by_id = {item["task"]["instance_id"]: item for item in items}
    heldout = [x for x in selection["ids"] if selection["splits"][x] == "heldout"]
    if len(heldout) != 8 or any(x not in sources for x in heldout):
        raise RuntimeError("heldout source cache incomplete")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    tokenizer, model = load_policy(config, device)
    actions = {}
    for task_id in heldout:
        prompt = agent_prompt(by_id[task_id], sources[task_id], "No edits yet.")
        actions[task_id] = [retokenized_action(
            tokenizer, prompt, row["completion"], config["context_tokens"])
            for row in grouped[task_id]]
    results = {"kind": "fixed_heldout_completion_ranking_not_solve_rate",
               "heldout_tasks": 8, "heldout_candidates": 128, "new_rollouts": 0,
               "new_score_calls": 0, "policies": {}}
    for policy in ("base", "test", "semantic"):
        if policy != "base":
            adapter_path = (args.training_dir / policy / "adapter_update_020" /
                            "adapter_model.safetensors")
            set_peft_model_state_dict(model, load_file(adapter_path))
        model.eval()
        logps = {}
        with torch.inference_mode():
            for index, task_id in enumerate(heldout, 1):
                logps[task_id] = [
                    float(completion_logp(model, action, device).mean())
                    for action in actions[task_id]]
                print(f"{policy} heldout task {index}/8", flush=True)
        targets = {}
        for target in ("terminal", "test", "semantic"):
            concordances = []
            total_c = total_d = 0
            for task_id in heldout:
                values = [float(row["score"]["solved"]) if target == "terminal"
                          else float(row[f"{target}_reward"]) for row in grouped[task_id]]
                c, d = pairwise(logps[task_id], values)
                total_c += c
                total_d += d
                if c + d:
                    concordances.append(c / (c + d))
            targets[target] = {
                "informative_tasks": len(concordances),
                "task_macro_concordance": mean(concordances) if concordances else None,
                "descriptive_pair_concordance": total_c / (total_c + total_d)
                if total_c + total_d else None,
                "comparable_pairs": total_c + total_d}
        results["policies"][policy] = targets
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
