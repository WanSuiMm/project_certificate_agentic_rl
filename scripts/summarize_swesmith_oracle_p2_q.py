#!/usr/bin/env python3
"""Compare frozen P1 scores and measured P2 q with first-hit continuation value.

Uses only the sanitized Oracle Credit export and the source-free P2 q export.
The four P2 observations per candidate are the first steps of the same four
continuations used for Q_hit; this is descriptive, not held-out validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from summarize_swesmith_oracle_credit import spearman_summary


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pair_accuracy(panel: list[dict], score: str, *, public_ties: bool = False) -> dict:
    correct = tied = total = 0
    for left, right in combinations(panel, 2):
        if left["Q_hit"] == right["Q_hit"]:
            continue
        if public_ties and left["p1"] != right["p1"]:
            continue
        total += 1
        direction = (left[score] - right[score]) * (left["Q_hit"] - right["Q_hit"])
        correct += direction > 0
        tied += direction == 0
    return {"oracle_nontied_pairs": total, "correct": correct, "proxy_ties": tied,
            "accuracy_ties_half": (correct + 0.5 * tied) / total if total else None}


def summarize(oracle_dir: Path, q2_path: Path) -> dict:
    old = json.loads((oracle_dir / "summary.json").read_text(encoding="utf-8"))
    if not old["completeness"]["data_complete"]:
        raise ValueError("Oracle Credit source is incomplete")
    candidates = {(task, row["sample"]): row
                  for task, group in old["tasks"].items() for row in group["candidates"]}
    states = {}
    for row in jsonl(oracle_dir / "states_public.jsonl"):
        key = row["instance_id"], row["sample"], row["replicate"], row["step"]
        if key in states:
            raise ValueError(f"duplicate public state: {key}")
        states[key] = row
    observations = {}
    for row in jsonl(q2_path):
        key = row["instance_id"], row["sample"], row["replicate"]
        if key in observations or row["step"] != 2:
            raise ValueError(f"duplicate or non-P2 observation: {key}")
        public = states.get((*key, 2))
        if (public is None or public["source_sha256"] != row["source_sha256"]
                or public["p_T"] != row["p2"] or public["solved"] != row["solved_p2"]):
            raise ValueError(f"P2 q is not aligned with frozen public state: {key}")
        observations[key] = row
    if len(candidates) != 96 or len(states) != 2688 or len(observations) != 384:
        raise ValueError("expected 96 P1 candidates, 2688 public states and 384 P2 observations")

    first_hits = Counter()
    panels: dict[str, list[dict]] = {task: [] for task in old["tasks"]}
    for (task, sample), candidate in sorted(candidates.items()):
        q2 = []
        hits = 0
        for replicate in range(4):
            key = task, sample, replicate
            if key not in observations or any((*key, step) not in states for step in range(2, 9)):
                raise ValueError(f"incomplete continuation: {key}")
            q2.append(observations[key]["q2"])
            hit = 1 if candidate["solved"] else next(
                (step for step in range(2, 9) if states[(*key, step)]["solved"]), None
            )
            if hit is not None:
                hits += 1
                first_hits[hit] += 1
        q1 = candidate["q"]
        mean_q2 = mean(q2) if all(q is not None for q in q2) else None
        panels[task].append({"sample": sample, "p1": candidate["p"], "q1": q1,
                             "P1_solved": candidate["solved"],
                             "P1_invalid_body": candidate["invalid_body"], "Q_hit": hits / 4,
                             "mean_q2": mean_q2,
                             "mean_delta_q": mean_q2 - q1 if mean_q2 is not None and q1 is not None else None,
                             "valid_q2_replicates": sum(q is not None for q in q2)})

    score_names = ("p1", "q1", "mean_q2", "mean_delta_q")
    task_metrics = {}
    for task, panel in panels.items():
        matched = [row for row in panel if all(row[score] is not None for score in score_names)]
        unsolved = [row for row in matched if not row["P1_solved"] and not row["P1_invalid_body"]]
        metrics = {}
        for score in score_names:
            metrics[score] = {
                "spearman_vs_Q_hit": spearman_summary(
                    [row[score] for row in matched], [row["Q_hit"] for row in matched]
                )["rho"],
                "pairwise": pair_accuracy(matched, score),
                "P1_unsolved_public_ties": pair_accuracy(unsolved, score, public_ties=True),
            }
        task_metrics[task] = {"candidates": len(panel), "matched_candidates": len(matched),
                              "matched_P1_unsolved_valid_body": len(unsolved),
                              "P1_candidates_with_any_first_hit": sum(row["Q_hit"] > 0 for row in panel),
                              "first_hits": int(sum(4 * row["Q_hit"] for row in panel)),
                              "metrics": metrics}
    informative = [task for task, row in task_metrics.items()
                   if any(row["metrics"][score]["spearman_vs_Q_hit"] is not None
                          for score in score_names)]
    return {"schema_version": 1, "target": "first solve at any P1..P8; absorbing-success recount",
            "first_hit_trajectories": sum(first_hits.values()),
            "first_hit_by_step": {str(step): first_hits[step] for step in range(1, 9)},
            "P2_q_valid_observations": sum(row["q2"] is not None for row in observations.values()),
            "P2_q_observations": len(observations),
            "fully_matched_candidates": sum(row["matched_candidates"] for row in task_metrics.values()),
            "informative_tasks": len(informative), "tasks": task_metrics,
            "task_macro_spearman": {
                score: mean(values) if (values := [task_metrics[task]["metrics"][score]["spearman_vs_Q_hit"]
                                                 for task in informative
                                                 if task_metrics[task]["metrics"][score]["spearman_vs_Q_hit"] is not None]) else None
                for score in score_names},
            "caveat": "K=4 first-hit Q is noisy; P2 q is computed from the same continuations as Q_hit, so this is not an independent predictive validation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-public-dir", type=Path, required=True)
    parser.add_argument("--p2-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.oracle_public_dir, args.p2_observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"first-hit={result['first_hit_trajectories']}; P2-q={result['P2_q_valid_observations']}/{result['P2_q_observations']}; matched={result['fully_matched_candidates']}")


if __name__ == "__main__":
    main()
