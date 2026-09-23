#!/usr/bin/env python3
"""Eight-edit on-policy SWE-smith GRPO: Test vs Semantic terminal credit.

Optional stepwise shaping is deliberately gated behind a completed pilot
summary. No frozen P1 candidates are used in this trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import pvariance

import torch

from capture_swesmith_body_trajectories import sample_step
from continue_swesmith_body_trajectories import public_fields, sha256
from grpo_objective import group_advantages
from swesmith_agent_edit import current_callable_body
from swesmith_modal_executor import SWESmithModalExecutor
from train_oracle_proxy_grpo import Action, cache_old_and_reference, update_policy
from train_swesmith_agent_grpo import append, load_policy, load_selected


@dataclass
class Trajectory:
    states: list[dict]
    actions: list[Action]


def trajectory_reward(solved: bool, potential: float, beta: float = 0.5) -> float:
    if not isinstance(potential, (int, float)) or not 0 <= potential <= 1:
        raise ValueError("missing or invalid terminal potential")
    return float(solved) + beta * float(potential)


def stepwise_returns(solved: bool, potentials: list[float],
                     beta: float = 0.5) -> list[float]:
    """Gamma=1 potential shaping: G_t=Y_8+beta*(phi_8-phi_{t-1})."""
    if len(potentials) != 9 or any(not isinstance(x, (int, float)) or not 0 <= x <= 1
                                    for x in potentials):
        raise ValueError("stepwise credit requires nine valid potentials")
    return [float(solved) + beta * (potentials[8] - potentials[t - 1])
            for t in range(1, 9)]


def stepwise_group_advantages(returns: list[list[float]], eps: float = 1e-6) -> list[list[float]]:
    """Normalize across trajectories separately at each action position."""
    if len(returns) < 2 or any(len(row) != 8 for row in returns):
        raise ValueError("stepwise advantages require a complete trajectory group")
    tensor = torch.tensor(returns, dtype=torch.float32)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False)
    advantage = torch.where(std > eps, (tensor - mean) / std.clamp_min(eps),
                            torch.zeros_like(tensor))
    return advantage.tolist()


def seed_for(seed: int, task_id: str, update: int, sample: int, step: int) -> int:
    return int(hashlib.sha256(f"{seed}|{task_id}|{update}|{sample}|{step}".encode()).hexdigest()[:8], 16)


def rollout_group(item: dict, model, tokenizer, executor, *, config: dict,
                  device: torch.device, update: int, group_size: int,
                  sample: bool, q_scope: str) -> list[Trajectory]:
    """Sample complete trajectories before measuring any hidden q value."""
    if q_scope not in {"none", "terminal", "all"}:
        raise ValueError("wrong q scope")
    task = item["task"]
    task_id = task["instance_id"]
    path = item["q_bank"]["callable"]
    with executor.task_session(task) as session:
        initialized = session.initialize(mode="init")
        if not initialized["gold_self_consistent"]:
            raise RuntimeError(f"gold self-consistency failed: {task_id}")
        source = initialized["buggy_source"]
        baseline = public_fields(session.call(mode="score", source=source,
                                              include_proxy=False))
        p0 = {"source": source, "source_sha256": sha256(source),
              "body": current_callable_body(source, path),
              "invalid_body": False, "feedback_given": None, **baseline}
        trajectories = []
        for sample_index in range(group_size):
            states, actions = [p0], []
            for step in range(1, 9):
                state, action = sample_step(
                    item=item, prior=states[-1], path=path, model=model,
                    tokenizer=tokenizer, session=session, config=config,
                    device=device, seed=seed_for(config["seed"], task_id, update,
                                                 sample_index, step), sample=sample,
                )
                states.append(state)
                actions.append(action)
            trajectories.append(Trajectory(states, actions))
        if q_scope != "none":
            session.set_q_bank(item["q_bank"])
            q_cache = {}
            for episode in trajectories:
                measured_states = episode.states[8:9] if q_scope == "terminal" else episode.states
                for state in measured_states:
                    key = state["source_sha256"]
                    if key not in q_cache:
                        result = session.call(mode="proxy_only", source=state["source"])
                        if result["status"] != "q_scored" or result.get("q") is None:
                            raise RuntimeError(f"invalid q observation: {task_id}/{key}")
                        q_cache[key] = result["q"]
                    state["q"] = q_cache[key]
        return trajectories


def action_advantages(episodes: list[Trajectory], arm: str,
                      credit: str, beta: float) -> tuple[list[float], list[float]]:
    """One group scalar per whole trajectory, or per-position shaped returns."""
    if len(episodes) != 16 or any(len(ep.actions) != 8 or len(ep.states) != 9 for ep in episodes):
        raise ValueError("expected 16 full eight-edit trajectories")
    terminal_rewards = []
    returns = []
    for ep in episodes:
        states = ep.states
        potentials = [float(row["p_T"] if arm == "test" else row["q"])
                      for row in (states if credit == "stepwise" else states[8:9])]
        terminal_rewards.append(trajectory_reward(states[8]["solved"], potentials[-1], beta))
        if credit == "stepwise":
            returns.append(stepwise_returns(states[8]["solved"], potentials, beta))
    if credit == "trajectory":
        group = group_advantages(torch.tensor(terminal_rewards), group_size=16).tolist()
        return terminal_rewards, [adv for adv in group for _ in range(8)]
    if credit == "stepwise":
        grouped = stepwise_group_advantages(returns)
        return terminal_rewards, [adv for row in grouped for adv in row]
    raise ValueError("wrong credit mode")


def evaluate(items: list[dict], model, tokenizer, executor, *, config: dict,
             device: torch.device, update: int) -> dict:
    details = []
    for item in items:
        ep = rollout_group(item, model, tokenizer, executor, config=config,
                           device=device, update=update, group_size=1,
                           sample=False, q_scope="none")[0]
        details.append({"instance_id": item["task"]["instance_id"],
                        "solved": ep.states[8]["solved"],
                        "p8": ep.states[8]["p_T"],
                        "source_sha256": ep.states[8]["source_sha256"]})
    return {"solved": sum(row["solved"] for row in details), "total": len(details),
            "public_test_solve_at_8": sum(row["solved"] for row in details) / len(details),
            "tasks": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "tasks", "q-results", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("test", "semantic"))
    parser.add_argument("--credit", choices=("trajectory", "stepwise"), default="trajectory")
    parser.add_argument("--pilot-summary", type=Path,
                        help="required for opt-in stepwise extension")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    if (config.get("protocol") != "swesmith_long_horizon_grpo_v01"
            or config["group_size"] != 16 or config["steps"] != 8
            or selected.get("status") != "q_first_frozen" or selected.get("count") != 28):
        raise ValueError("wrong frozen long-horizon protocol")
    if args.credit == "stepwise":
        if args.pilot_summary is None:
            raise ValueError("stepwise extension requires explicit pilot summary")
        pilot = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
        if pilot.get("kind") != "frozen_policy_trajectory_grpo_reward_resolution":
            raise ValueError("wrong pilot summary")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    items = load_selected(args.tasks, args.q_results, selected["ids"])
    train = [item for item in items if selected["splits"][item["task"]["instance_id"]] == "train"]
    heldout = [item for item in items if selected["splits"][item["task"]["instance_id"]] == "heldout"]
    if len(train) != config["train_tasks"] or len(heldout) != config["heldout_tasks"]:
        raise ValueError("frozen split is not 20 train / 8 held-out tasks")
    tokenizer, model = load_policy(config, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["learning_rate"])
    executor = SWESmithModalExecutor()
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    rollouts_path = args.output_dir / "rollouts.jsonl"
    append(metrics_path, {"event": "setup", "arm": args.arm, "credit": args.credit,
           "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
           "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "model_revision": config["model_revision"],
           "time_utc": datetime.now(timezone.utc).isoformat()})
    append(metrics_path, {"event": "eval", "update": 0,
           **evaluate(heldout, model, tokenizer, executor,
                      config=config, device=device, update=0)})
    for update in range(1, config["updates_per_arm"] + 1):
        item = train[(update - 1) % len(train)]
        task_id = item["task"]["instance_id"]
        q_scope = ("none" if args.arm == "test" else
                   "terminal" if args.credit == "trajectory" else "all")
        episodes = rollout_group(item, model, tokenizer, executor, config=config,
                                 device=device, update=update, group_size=16,
                                 sample=True, q_scope=q_scope)
        rewards, advantages = action_advantages(episodes, args.arm, args.credit,
                                                config["beta"])
        for sample_index, ep in enumerate(episodes):
            append(rollouts_path, {
                "update": update, "instance_id": task_id, "trajectory_id": sample_index,
                "states": [{"step": step, "source_sha256": state["source_sha256"],
                            "p_T": state["p_T"], "solved": state["solved"],
                            "q": state.get("q"), "invalid_body": state.get("invalid_body")}
                           for step, state in enumerate(ep.states)],
                "completions": [action.completion for action in ep.actions],
                "token_ids": [action.token_ids.tolist() for action in ep.actions],
                "reward": rewards[sample_index],
            })
        info = {"event": "train", "update": update, "task_id": task_id,
                "arm": args.arm, "credit": args.credit,
                "reward_variance": pvariance(rewards),
                "terminal_solved": sum(ep.states[8]["solved"] for ep in episodes),
                "nonzero_advantages": sum(abs(value) > 0 for value in advantages),
                "environment_steps": 16 * 8}
        if not any(abs(value) > 0 for value in advantages):
            info.update({"zero_advantage_group": True, "optimizer_step": False,
                         "grad_norm": 0.0})
        else:
            actions = [action for ep in episodes for action in ep.actions]
            cache_old_and_reference(model, actions, device)
            info.update(update_policy(model, optimizer, actions, advantages,
                                      device=device, config=config))
            info.update({"zero_advantage_group": False, "optimizer_step": True})
        append(metrics_path, info)
        if update % 5 == 0 or update == config["updates_per_arm"]:
            model.save_pretrained(args.output_dir / f"adapter_update_{update:03d}")
        if update % config["eval_every_updates"] == 0 or update == config["updates_per_arm"]:
            append(metrics_path, {"event": "eval", "update": update,
                   **evaluate(heldout, model, tokenizer, executor,
                              config=config, device=device, update=update)})
        print(f"{args.arm}/{args.credit} update {update}/{config['updates_per_arm']}", flush=True)
    append(metrics_path, {"event": "complete", "time_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
