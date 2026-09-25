#!/usr/bin/env python3
"""Frozen P1 candidates -> repeated public-feedback continuations -> MC value.

No optimizer, LoRA update, or P2..P8 semantic grading. Candidate code executes
only in network-blocked Modal sandboxes. Completed edits and pending sampled
actions are durable, so a restart does not resample successful generation.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import time

from continue_swesmith_body_trajectories import feedback_from_public, public_fields, sha256
from swesmith_agent_edit import InvalidEdit, replace_callable_body
from swesmith_modal_executor import SWESmithModalExecutor, WorkerEndedError
from train_swesmith_agent_grpo import agent_prompt, load_selected


SYSTEM_PROMPT = "You repair Python functions. Output only the replacement function body, without def or explanation."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Fail on incomplete writes rather than silently dropping evidence.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, row: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_key(row: dict) -> tuple:
    return row["instance_id"], row["sample"]


def state_key(row: dict) -> tuple:
    return (*candidate_key(row), row["replicate"], row["step"])


def unique_rows(rows: list[dict], key_function) -> dict:
    indexed = {}
    for row in rows:
        key = key_function(row)
        if key in indexed:
            raise ValueError(f"duplicate record: {key}")
        if "source" in row and sha256(row["source"]) != row["source_sha256"]:
            raise ValueError(f"source/hash mismatch: {key}")
        indexed[key] = row
    return indexed


def freeze_candidates(selection: dict, census: list[dict], config: dict) -> tuple[list[str], list[dict]]:
    if (selection.get("status") != "q_first_frozen" or selection.get("count") != 28
            or len(selection["ids"]) != 28 or len(set(selection["ids"])) != 28):
        raise ValueError("expected frozen 28-task selection")
    ids = selection["ids"][:config["pilot_tasks"]]
    indexed = unique_rows(census, candidate_key)
    selected = []
    for task_id in ids:
        for sample in range(config["candidates_per_task"]):
            row = indexed.get((task_id, sample))
            if row is None or row.get("status") not in {"scored", "invalid_body", "invalid_candidate"}:
                raise ValueError(f"missing/errored frozen candidate: {task_id}/{sample}")
            if "completion" not in row or "final_source_sha256" not in row:
                raise ValueError("cannot reconstruct frozen candidate")
            selected.append(row)
    return ids, selected


def reconstruct_p1(buggy: str, path: list[str], row: dict) -> tuple[str, bool]:
    try:
        source, invalid = replace_callable_body(buggy, path, row["completion"]), False
    except InvalidEdit:
        source, invalid = buggy, True
    if invalid != (row["status"] == "invalid_body") or sha256(source) != row["final_source_sha256"]:
        raise ValueError(f"frozen P1 reconstruction mismatch: {candidate_key(row)}")
    return source, invalid


def public_item(item: dict) -> dict:
    return {"task": item["task"], "q_bank": {
        key: item["q_bank"][key] for key in ("module", "callable")}}


def batch_seed(config: dict, keys: list[tuple]) -> int:
    value = json.dumps([config["seed"], keys], separators=(",", ":"))
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def load_frozen_policy(config: dict):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    snapshot = snapshot_download(config["model"], revision=config["model_revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.bfloat16,
        trust_remote_code=False, attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval().requires_grad_(False)
    model.config.use_cache = True
    return tokenizer, model


def generate_batch(model, tokenizer, prompts: list[str], *, config: dict, seed: int) -> list[dict]:
    import torch

    messages = [[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}] for prompt in prompts]
    text = [tokenizer.apply_chat_template(row, tokenize=False, add_generation_prompt=True)
            for row in messages]
    encoded = tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    width = encoded["input_ids"].shape[1]
    if width + config["max_new_tokens"] > config["context_tokens"]:
        raise RuntimeError("prompt exceeds frozen context budget")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=True, use_cache=True,
            max_new_tokens=config["max_new_tokens"], temperature=config["temperature"],
            top_p=config["top_p"], top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for tokens in generated[:, width:].cpu().tolist():
        ended = tokenizer.eos_token_id in tokens
        if ended:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
        results.append({"completion": tokenizer.decode(tokens, skip_special_tokens=True),
                        "generated_tokens": len(tokens), "hit_token_cap": not ended})
    return results


def prepare_candidates(item: dict, rows: list[dict], session, initialized: dict) -> list[dict]:
    """Verify P1 hashes/public measurements; repair artificial invalid-edit zeros."""
    task_id = item["task"]["instance_id"]
    buggy, path = initialized["buggy_source"], item["q_bank"]["callable"]
    observed_public, observed_q = {}, {}
    bank_set = False
    candidates = []
    for row in rows:
        source, invalid = reconstruct_p1(buggy, path, row)
        key = sha256(source)
        if key not in observed_public:
            observed_public[key] = public_fields(session.call(mode="score", source=source, include_proxy=False))
        actual = observed_public[key]
        old = row["score"]
        if row["status"] == "scored" and (actual["p_T"] != old["p_T"] or actual["solved"] != old["solved"]):
            raise RuntimeError(f"frozen P1 public result drift: {task_id}/{row['sample']}")
        if invalid:
            if key not in observed_q:
                if not bank_set:
                    session.set_q_bank(item["q_bank"])
                    bank_set = True
                measured = session.call(mode="proxy_only", source=source)
                observed_q[key] = measured["q"] if measured["status"] == "q_scored" else None
            q = observed_q[key]
            origin = "actual_unchanged_P0_remeasured"
        else:
            q = old.get("q") if row["status"] == "scored" and not old.get("q_invalid_candidate") else None
            origin = "frozen_census" if q is not None else "invalid_proxy_observation"
        if q is not None and (not isinstance(q, (float, int)) or not 0 <= q <= 1):
            raise ValueError("invalid frozen q")
        candidates.append({"instance_id": task_id, "sample": row["sample"],
                           "source": source, "source_sha256": key,
                           "p": actual["p_T"], "q": q, "proxy_valid": q is not None,
                           "q_origin": origin, "invalid_body": invalid, **actual})
    return candidates


def score_pending(pending: dict, sessions: Queue, executor, task: dict) -> dict:
    session = sessions.get()
    started = time.monotonic()
    try:
        if session is None:
            result = executor.call(task, mode="score", source=pending["source"], include_proxy=False)
        else:
            try:
                result = session.call(mode="score", source=pending["source"], include_proxy=False)
            except WorkerEndedError:
                # The failed persistent worker must never be handed to another action.
                # Retry this exact frozen source once in an independently initialized sandbox.
                session = None
                result = executor.call(task, mode="score", source=pending["source"], include_proxy=False)
        public = public_fields(result)
    finally:
        sessions.put(session)
    # q has never entered pending actions or public observations.
    return {**pending, **public, "public_seconds": time.monotonic() - started}


def validate_state_chain(candidates: dict, actions: dict, states: dict, outcomes: dict) -> None:
    for key, state in states.items():
        if key not in actions or any(state[field] != actions[key][field]
                                     for field in ("source_sha256", "prior_source_sha256", "feedback_given")):
            raise ValueError(f"saved state lacks matching sampled action: {key}")
    for key, action in actions.items():
        task_id, sample, replicate, step = key
        prior = candidates.get((task_id, sample)) if step == 2 else states.get((task_id, sample, replicate, step - 1))
        if prior is None or action["prior_source_sha256"] != prior["source_sha256"]:
            raise ValueError(f"broken source-state chain: {key}")
        if action["feedback_given"] != feedback_from_public(prior):
            raise ValueError(f"broken public-feedback chain: {key}")
    for key, outcome in outcomes.items():
        final = states.get((*key, 8))
        if final is None or outcome["source_sha256"] != final["source_sha256"] or outcome["solved"] != final["solved"]:
            raise ValueError(f"outcome lacks matching P8: {key}")


def import_checkpoint(source: Path, output: Path, receipt: dict) -> dict:
    """Copy a failed run's durable journal into a new code-versioned run."""
    source = source.resolve(strict=True)
    if source == output.resolve() or any(output.glob("*.jsonl")) or any((output / "action_batches").iterdir()):
        raise ValueError("checkpoint destination is not empty")
    previous = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if previous["status"] not in {"failed", "interrupted"}:
        raise ValueError("only a failed or explicitly interrupted run may be imported")
    for key in ("config", "task_ids", "selection_sha256", "census_sha256", "tasks_sha256", "q_results_sha256"):
        if previous[key] != receipt[key]:
            raise ValueError(f"checkpoint input changed: {key}")
    files = ["frozen_p1_inputs.jsonl", "candidates.jsonl", "states.jsonl", "outcomes.jsonl"]
    batches = sorted((source / "action_batches").glob("*.json"))
    if not batches:
        raise ValueError("checkpoint has no sampled actions")
    evidence_hashes = {}
    for name in files:
        path = source / name
        evidence_hashes[name] = digest(path)
        shutil.copy2(path, output / name)
    for path in batches:
        evidence_hashes[f"action_batches/{path.name}"] = digest(path)
        shutil.copy2(path, output / "action_batches" / path.name)
    imported = {"source_run": str(source), "source_run_sha256": digest(source / "run.json"),
                "source_code_sha256": previous["code_sha256"],
                "completed_edit_steps": previous["completed_edit_steps"],
                "completed_endpoints": previous["completed_endpoints"],
                "evidence_sha256": evidence_hashes, "imported_utc": now()}
    write_json(output / "import_manifest.json", imported)
    return {key: imported[key] for key in ("source_run_sha256", "completed_edit_steps", "completed_endpoints")}


def run_task(item: dict, frozen: list[dict], *, config: dict, output: Path, executor,
             tokenizer, model, candidates: dict, actions: dict, states: dict, outcomes: dict,
             progress) -> None:
    task_id = item["task"]["instance_id"]
    ids = [(task_id, sample, replicate) for sample in range(16)
           for replicate in range(config["continuations_per_candidate"])]
    if all(key in outcomes for key in ids):
        return
    with ExitStack() as stack:
        first = stack.enter_context(executor.task_session(item["task"]))
        saved_candidates = all((task_id, sample) in candidates for sample in range(16))
        init = first.initialize(mode="source_only" if saved_candidates else "init")
        if saved_candidates:
            # This task passed gold and P1 public checks before its candidate
            # journal was saved; verify P1 reconstruction without paying again.
            for row in frozen:
                source, invalid = reconstruct_p1(init["buggy_source"], item["q_bank"]["callable"], row)
                saved = candidates[candidate_key(row)]
                if saved["source_sha256"] != sha256(source) or saved["invalid_body"] != invalid:
                    raise RuntimeError(f"candidate changed on resume: {candidate_key(row)}")
        else:
            if not init["gold_self_consistent"]:
                raise RuntimeError(f"gold self-consistency failed: {task_id}")
            for row in prepare_candidates(item, frozen, first, init):
                key = candidate_key(row)
                if key in candidates:
                    for field in ("source_sha256", "p", "q", "solved", "proxy_valid", "invalid_body"):
                        if candidates[key][field] != row[field]:
                            raise RuntimeError(f"candidate changed on resume: {key}/{field}")
                else:
                    append_row(output / "candidates.jsonl", row)
                    candidates[key] = row
        pool = Queue()
        pool.put(first)
        for _ in range(config["sandbox_workers"] - 1):
            session = stack.enter_context(executor.task_session(item["task"]))
            replica = session.initialize(mode="source_only")
            if replica["buggy_source_sha256"] != init["buggy_source_sha256"]:
                raise RuntimeError("sandbox initial source mismatch")
            pool.put(session)
        with ThreadPoolExecutor(max_workers=config["sandbox_workers"]) as scorer:
            ready = deque()
            futures = {}

            def submit_score(key):
                futures[scorer.submit(score_pending, actions[key], pool, executor, item["task"])] = key

            # Each unfinished trajectory contributes only its earliest missing
            # state. Earlier saved actions are never sampled again.
            for trajectory in ids:
                if trajectory in outcomes:
                    continue
                for step in range(2, 9):
                    key = (*trajectory, step)
                    if key in states:
                        continue
                    if key in actions:
                        submit_score(key)
                    else:
                        ready.append(key)
                    break

            flush_partial = False
            while ready or futures:
                completed = [future for future in futures if future.done()]
                for future in completed:
                    key = futures.pop(future)
                    row = future.result()
                    if state_key(row) != key:
                        raise RuntimeError("scorer returned another action")
                    append_row(output / "states.jsonl", row)
                    states[key] = row
                    if key[3] == 8 and key[:3] not in outcomes:
                        endpoint = {"instance_id": task_id, "sample": row["sample"], "replicate": row["replicate"],
                                    "solved": row["solved"], "p8": row["p_T"], "source_sha256": row["source_sha256"],
                                    "public_valid": row["public"].get("valid", False)}
                        append_row(output / "outcomes.jsonl", endpoint)
                        outcomes[key[:3]] = endpoint
                    elif key[3] < 8:
                        ready.append((*key[:3], key[3] + 1))
                    progress(task_id, key[3])
                if completed:
                    continue

                if ready and (len(ready) >= config["generation_batch_size"] or not futures or flush_partial):
                    flush_partial = False
                    chunk = [ready.popleft() for _ in range(min(len(ready), config["generation_batch_size"]))]
                    priors = [candidates[key[:2]] if key[3] == 2 else states[(*key[:3], key[3] - 1)] for key in chunk]
                    prompts = [agent_prompt(public_item(item), row["source"], feedback_from_public(row)) for row in priors]
                    started = time.monotonic()
                    seed = batch_seed(config, chunk)
                    completions = generate_batch(model, tokenizer, prompts, config=config, seed=seed)
                    if len(completions) != len(chunk):
                        raise RuntimeError("generation batch size mismatch")
                    batch_records = []
                    for key, prior, completion in zip(chunk, priors, completions):
                        try:
                            source = replace_callable_body(prior["source"], item["q_bank"]["callable"], completion["completion"])
                            invalid = False
                        except InvalidEdit:
                            source, invalid = prior["source"], True
                        batch_records.append({"instance_id": key[0], "sample": key[1], "replicate": key[2], "step": key[3],
                                              "source": source, "source_sha256": sha256(source),
                                              "prior_source_sha256": prior["source_sha256"],
                                              "feedback_given": feedback_from_public(prior), "invalid_body": invalid,
                                              "batch_seed": seed, "generation_batch_seconds": time.monotonic() - started,
                                              **completion})
                    batch_name = hashlib.sha256(json.dumps(chunk).encode()).hexdigest()[:20] + ".json"
                    write_json(output / "action_batches" / batch_name, {"keys": chunk, "records": batch_records})
                    for row in batch_records:
                        key = state_key(row)
                        actions[key] = row
                        submit_score(key)
                    continue

                if futures:
                    done, _ = wait(tuple(futures), timeout=0.5, return_when=FIRST_COMPLETED)
                    # Send a partial ready batch if the scorers do not return
                    # soon; a single slow test must not be a step barrier.
                    flush_partial = bool(ready) and not done

            print(json.dumps({"event": "task_complete", "task": task_id,
                              "states": len(states), "endpoints": len(outcomes), "time": now()}), flush=True)
    # Recover the tiny gap between the last state append and outcome append.
    for key in ids:
        if key not in outcomes and (*key, 8) in states:
            row = states[(*key, 8)]
            endpoint = {"instance_id": task_id, "sample": key[1], "replicate": key[2],
                        "solved": row["solved"], "p8": row["p_T"], "source_sha256": row["source_sha256"],
                        "public_valid": row["public"].get("valid", False)}
            append_row(output / "outcomes.jsonl", endpoint)
            outcomes[key] = endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "census", "tasks", "q-results", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--import-run-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("protocol") != "swesmith_oracle_credit_v01" or config["pilot_tasks"] != 6
            or config["candidates_per_task"] != 16 or config["continuations_per_candidate"] != 4
            or config["terminal_step"] != 8 or config["max_new_tokens"] != 1024):
        raise ValueError("wrong frozen 6x16x4 pilot")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    task_ids, frozen = freeze_candidates(selection, read_rows(args.census), config)
    receipt = {"kind": "frozen_policy_mc_continuation_value", "status": "running",
               "task_ids": task_ids, "candidates_per_task": 16, "continuations_per_candidate": 4,
               "terminal_step": 8, "expected_endpoints": 384, "expected_edit_steps": 2688,
               "config": config, "selection_sha256": digest(args.selection), "census_sha256": digest(args.census),
               "tasks_sha256": digest(args.tasks), "q_results_sha256": digest(args.q_results),
               "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "started_utc": now(), "success_definition": "all selected official public tests pass at P8"}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "action_batches").mkdir(exist_ok=True)
    run_path = output / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key in ("config", "task_ids", "selection_sha256", "census_sha256", "tasks_sha256", "q_results_sha256", "code_sha256"):
            if previous[key] != receipt[key]:
                raise ValueError(f"changed resume inputs: {key}")
        if previous["status"] == "complete":
            print("already complete", flush=True)
            return
        receipt["started_utc"] = previous["started_utc"]
    elif any(output.glob("*.jsonl")):
        raise ValueError("unregistered output files")
    elif args.import_run_dir is not None:
        imported = import_checkpoint(args.import_run_dir, output, receipt)
        receipt["imported_checkpoint"] = imported
        receipt["completed_edit_steps"] = imported["completed_edit_steps"]
        receipt["completed_endpoints"] = imported["completed_endpoints"]
    write_json(run_path, receipt)
    frozen_path = output / "frozen_p1_inputs.jsonl"
    if frozen_path.exists():
        if read_rows(frozen_path) != frozen:
            raise ValueError("frozen P1 input changed")
    else:
        for row in frozen:
            append_row(frozen_path, row)
    candidates = unique_rows(read_rows(output / "candidates.jsonl"), candidate_key)
    states = unique_rows(read_rows(output / "states.jsonl"), state_key)
    actions = unique_rows([row for path in sorted((output / "action_batches").glob("*.json"))
                           for row in json.loads(path.read_text(encoding="utf-8"))["records"]], state_key)
    outcomes = unique_rows(read_rows(output / "outcomes.jsonl"), lambda row: (*candidate_key(row), row["replicate"]))
    validate_state_chain(candidates, actions, states, outcomes)

    def progress(task_id, step):
        receipt.update(current_task=task_id, current_step=step, completed_edit_steps=len(states),
                       completed_endpoints=len(outcomes), updated_utc=now())
        write_json(run_path, receipt)

    try:
        items = load_selected(args.tasks, args.q_results, task_ids)
        tokenizer, model = load_frozen_policy(config)
        receipt["runtime_versions"] = {}
        import importlib.metadata
        for package in ("torch", "transformers", "modal"):
            receipt["runtime_versions"][package] = importlib.metadata.version(package)
        write_json(run_path, receipt)
        executor = SWESmithModalExecutor()
        for item in items:
            block = [row for row in frozen if row["instance_id"] == item["task"]["instance_id"]]
            run_task(item, block, config=config, output=output, executor=executor,
                     tokenizer=tokenizer, model=model, candidates=candidates, actions=actions,
                     states=states, outcomes=outcomes, progress=progress)
        validate_state_chain(candidates, actions, states, outcomes)
        if len(states) != 2688 or len(outcomes) != 384 or len(candidates) != 96:
            raise RuntimeError("incomplete oracle benchmark")
        receipt.update(status="complete", completed_edit_steps=len(states), completed_endpoints=len(outcomes), finished_utc=now())
        write_json(run_path, receipt)
        from summarize_swesmith_oracle_credit import main as summarize_main
        import sys
        sys.argv = [sys.argv[0], "--run-dir", str(output)]
        summarize_main()
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000], updated_utc=now())
        write_json(run_path, receipt)
        raise


if __name__ == "__main__":
    main()
