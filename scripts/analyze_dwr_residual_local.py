#!/usr/bin/env python3
"""Frozen 8-bin residual audit against saved P2 continuation outcomes.

The 256 probe positions are split into eight contiguous 32-probe bins before
looking at outcomes. A fixed-alpha ridge model learns bin weights only from
other P1 candidates (or other tasks); evaluation uses mixed-outcome, valid-P2
replicates within the held-out P1 candidate. This is exploratory and is not a
new RL result or an oracle-value estimate.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "runs/oracle_credit_6x16x4_v01/run_v03"
VECTORS = PROJECT / "runs/dwr_residual_audit_v01/vectors.jsonl"
Q2 = PROJECT / "evidence/oracle_credit_p2_q_v01/observations.jsonl"
OUTCOMES = PROJECT / "evidence/oracle_credit_6x16x4_v01/outcomes_public.jsonl"
STATES_PUBLIC = PROJECT / "evidence/oracle_credit_6x16x4_v01/states_public.jsonl"
OUTPUT = PROJECT / "runs/dwr_residual_audit_v01/analysis.json"
PUBLIC_PANEL = PROJECT / "runs/dwr_residual_audit_v01/panel_public.jsonl"


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def blocks(bits: str, bins: int = 8) -> np.ndarray:
    if len(bits) != 256 or 256 % bins:
        raise ValueError("not a 256-probe vector")
    return np.fromiter((bits[i:i + 256 // bins].count("1") / (256 // bins)
                        for i in range(0, 256, 256 // bins)), dtype=float)


def load_panel() -> list[dict]:
    vectors = {(r["instance_id"], r["source_sha256"]): r for r in rows(VECTORS)}
    if not vectors or not all(r["parity"] and r["q_local"] == r["q_frozen"] for r in vectors.values()):
        raise ValueError("frozen-q parity incomplete")
    candidates = {(r["instance_id"], r["sample"]): r for r in rows(SOURCE / "candidates.jsonl")}
    p2q = {(r["instance_id"], r["sample"], r["replicate"]): r for r in rows(Q2)}
    future = {(r["instance_id"], r["sample"], r["replicate"]): r for r in rows(OUTCOMES)}
    future_hit = defaultdict(bool)
    for row in rows(STATES_PUBLIC):
        if row["step"] >= 3:
            key = row["instance_id"], row["sample"], row["replicate"]
            future_hit[key] |= bool(row["solved"])
    panel = []
    for state in rows(SOURCE / "states.jsonl"):
        if state["step"] != 2:
            continue
        task, sample, replicate = state["instance_id"], state["sample"], state["replicate"]
        key = task, sample, replicate
        p1 = candidates[task, sample]
        if task not in {t for t, _ in vectors}:
            continue
        if p1["solved"] or state["solved"] or state["invalid_body"]:
            continue
        if key not in p2q or p2q[key]["q2"] is None or key not in future:
            continue
        v1 = vectors.get((task, p1["source_sha256"]))
        v2 = vectors.get((task, state["source_sha256"]))
        if v2 is None:
            raise ValueError(f"missing vector for valid branch: {key}")
        if (v1 is not None and p1["q"] is not None and v1["q_local"] != p1["q"]) or v2["q_local"] != p2q[key]["q2"]:
            raise ValueError(f"scalar q mismatch for branch: {key}")
        panel.append({"task": task, "sample": sample, "replicate": replicate,
                      "source_sha256": state["source_sha256"],
                      "y": int(future_hit[key]), "endpoint_y": int(future[key]["solved"]),
                      "p": p2q[key]["p2"],
                      "q": v2["q_local"],
                      "b8": blocks(v2["match_bits"]),
                      "b16": blocks(v2["match_bits"], 16)})
    return panel


def save_public_panel(panel: list[dict]) -> None:
    PUBLIC_PANEL.parent.mkdir(parents=True, exist_ok=True)
    with PUBLIC_PANEL.open("w", encoding="utf-8", newline="\n") as handle:
        for row in panel:
            public = {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in row.items()}
            handle.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")


def load_public_panel() -> list[dict]:
    panel = list(rows(PUBLIC_PANEL))
    if not panel:
        raise ValueError("empty public panel")
    for row in panel:
        for key in ("b8", "b16"):
            row[key] = np.asarray(row[key], dtype=float)
        if len(row["b8"]) != 8 or len(row["b16"]) != 16:
            raise ValueError("invalid public panel bin width")
        if not np.isclose(np.mean(row["b8"]), row["q"]):
            raise ValueError("public panel q does not equal bin mean")
    return panel


def fit_scores(panel: list[dict], feature: str, holdout: str) -> list[float]:
    groups = defaultdict(list)
    for i, row in enumerate(panel):
        group = (row["task"], row["sample"]) if holdout == "candidate" else row["task"]
        groups[group].append(i)
    predictions = [float("nan")] * len(panel)
    for held, indices in groups.items():
        held_sources = {(panel[i]["task"], panel[i]["source_sha256"]) for i in indices}
        train = [i for i, row in enumerate(panel)
                 if ((row["task"], row["sample"]) if holdout == "candidate" else row["task"]) != held
                 and (row["task"], row["source_sha256"]) not in held_sources]
        if not train or len({panel[i]["y"] for i in train}) < 2:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(np.array([panel[i][feature] for i in train]),
                  np.array([panel[i]["y"] for i in train]))
        fitted = model.predict(np.array([panel[i][feature] for i in indices]))
        for i, value in zip(indices, fitted):
            predictions[i] = float(value)
    return predictions


def pair_accuracy(panel: list[dict], scores: list[float]) -> dict:
    groups = defaultdict(list)
    for i, row in enumerate(panel):
        groups[row["task"], row["sample"]].append(i)
    by_task = defaultdict(list)
    for (task, _), indices in groups.items():
        positives = [i for i in indices if panel[i]["y"]]
        negatives = [i for i in indices if not panel[i]["y"]]
        if not positives or not negatives:
            continue
        correct = ties = total = 0
        for i in positives:
            for j in negatives:
                if not np.isfinite(scores[i]) or not np.isfinite(scores[j]):
                    continue
                correct += scores[i] > scores[j]
                ties += scores[i] == scores[j]
                total += 1
        if total:
            by_task[task].append({"correct": int(correct), "ties": int(ties),
                                  "pairs": total, "accuracy": (correct + 0.5 * ties) / total})
    groups_flat = [g for task in by_task for g in by_task[task]]
    return {"mixed_candidates": len(groups_flat), "mixed_tasks": len(by_task),
            "pairs": sum(g["pairs"] for g in groups_flat),
            "correct": sum(g["correct"] for g in groups_flat),
            "ties": sum(g["ties"] for g in groups_flat),
            "candidate_macro_accuracy": float(np.mean([g["accuracy"] for g in groups_flat]))
            if groups_flat else None,
            "task_macro_accuracy": float(np.mean([np.mean([g["accuracy"] for g in groups])
                                              for groups in by_task.values()])) if by_task else None,
            "per_task": {task: {"groups": len(groups),
                                 "candidate_macro_accuracy": float(np.mean([g["accuracy"] for g in groups]))}
                         for task, groups in by_task.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-public-panel", action="store_true",
                        help="recompute from the uploaded source-free panel")
    args = parser.parse_args()
    panel = load_public_panel() if args.from_public_panel else load_panel()
    if not args.from_public_panel:
        save_public_panel(panel)
    summary = {"protocol": "8 contiguous frozen 32-probe bins; valid unsolved P2; target=first public-test hit in P3-P8; fixed ridge alpha=10",
               "source": str(PUBLIC_PANEL.relative_to(PROJECT)), "states": len(panel),
               "tasks": len({r["task"] for r in panel}), "future_first_hits": sum(r["y"] for r in panel),
               "endpoint_successes": sum(r["endpoint_y"] for r in panel),
               "q_parity_sources": sum(1 for _ in rows(VECTORS)), "metrics": {}}
    summary["metrics"]["public_p2"] = pair_accuracy(panel, [r["p"] for r in panel])
    summary["metrics"]["scalar_q2"] = pair_accuracy(panel, [r["q"] for r in panel])
    for holdout in ("candidate", "task"):
        for feature in ("b8", "b16"):
            name = f"{feature}_ridge_leave_one_{holdout}_out"
            summary["metrics"][name] = pair_accuracy(panel, fit_scores(panel, feature, holdout))
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"states": summary["states"], "future_first_hits": summary["future_first_hits"],
                      "endpoint_successes": summary["endpoint_successes"],
                      "metrics": {key: {k: value for k, value in metric.items()
                                         if k in ("mixed_candidates", "mixed_tasks", "pairs",
                                                  "candidate_macro_accuracy", "task_macro_accuracy")}
                                  for key, metric in summary["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
