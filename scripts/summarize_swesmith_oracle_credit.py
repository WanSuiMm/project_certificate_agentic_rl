"""Summarize repeated frozen-policy continuations for SWE-smith P1 candidates.

The repeated terminal solve rate Q is a noisy Monte Carlo target.  The
summarizer reports descriptive within-task rank and pairwise comparisons for p
and q; it does not treat the repeated outcomes as ground truth or make a
KEEP/KILL decision.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Callable


WILSON_Z_95 = 1.959963984540054


def _required_string(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value


def _required_int(row: dict[str, Any], key: str, where: str, *, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where}: {key} must be an integer >= {minimum}")
    return value


def _required_bool(row: dict[str, Any], key: str, where: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{where}: {key} must be a boolean")
    return value


def _probability(value: Any, key: str, where: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{where}: {key} must be a finite number in [0, 1]")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.is_file():
        return [], False
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        row["_line_number"] = line_number
        rows.append(row)
    return rows, True


def wilson_interval(successes: int, trials: int) -> list[float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def average_ranks(values: list[float]) -> list[float]:
    """Compute one-based average ranks, assigning equal values tied ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = math.fsum(value * value for value in left_centered)
    right_ss = math.fsum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    covariance = math.fsum(a * b for a, b in zip(left_centered, right_centered))
    return covariance / math.sqrt(left_ss * right_ss)


def spearman_summary(proxy: list[float], q_values: list[float]) -> dict[str, Any]:
    if len(proxy) != len(q_values):
        raise ValueError("Spearman inputs must have equal lengths")
    rho = _pearson_correlation(average_ranks(proxy), average_ranks(q_values))
    reason = None
    if rho is None:
        if len(proxy) < 2:
            reason = "fewer_than_two_observations"
        elif len(set(q_values)) == 1:
            reason = "constant_Q"
        elif len(set(proxy)) == 1:
            reason = "constant_proxy"
        else:
            reason = "undefined_correlation"
    return {"n": len(proxy), "rho": rho, "null_reason": reason}


def _intervals_disjoint(left: list[float], right: list[float]) -> bool:
    return left[1] < right[0] or right[1] < left[0]


def _pairwise_metrics(
    candidates: list[dict[str, Any]],
    score_key: str,
    pair_filter: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    empirical_pairs = 0
    scored_pairs = 0
    empirical_correct = 0.0
    confident_pairs = 0
    confident_scored_pairs = 0
    confident_correct = 0.0

    for left, right in combinations(candidates, 2):
        if pair_filter is not None and not pair_filter(left, right):
            continue
        if left["oracle_successes"] == right["oracle_successes"]:
            continue

        empirical_pairs += 1
        left_score = left[score_key]
        right_score = right[score_key]
        if left_score is not None and right_score is not None:
            scored_pairs += 1
            if left_score == right_score:
                empirical_correct += 0.5
            elif (left_score > right_score) == (
                left["oracle_successes"] > right["oracle_successes"]
            ):
                empirical_correct += 1.0

        is_confident = _intervals_disjoint(left["wilson95"], right["wilson95"])
        if is_confident:
            confident_pairs += 1
            if left_score is not None and right_score is not None:
                confident_scored_pairs += 1
                if left_score == right_score:
                    confident_correct += 0.5
                elif (left_score > right_score) == (
                    left["oracle_successes"] > right["oracle_successes"]
                ):
                    confident_correct += 1.0

    return {
        "proxy_ties_score_half": True,
        "empirical_all_pair": {
            "non_tied_Q_pairs": empirical_pairs,
            "scored_pairs": scored_pairs,
            "coverage": scored_pairs / empirical_pairs if empirical_pairs else None,
            "accuracy": empirical_correct / scored_pairs if scored_pairs else None,
        },
        "confident_pair_direction": {
            "eligible_pairs": confident_pairs,
            "scored_pairs": confident_scored_pairs,
            "coverage_among_confident_pairs": (
                confident_scored_pairs / confident_pairs if confident_pairs else None
            ),
            "coverage_of_empirical_non_tied_pairs": (
                confident_pairs / empirical_pairs if empirical_pairs else None
            ),
            "accuracy": (
                confident_correct / confident_scored_pairs if confident_scored_pairs else None
            ),
        },
    }


def _matched_pairwise_metrics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare p and q on one q-valid candidate and candidate-pair support."""
    p_metrics = _pairwise_metrics(candidates, "p")
    q_metrics = _pairwise_metrics(candidates, "q")
    p_empirical = p_metrics["empirical_all_pair"]
    q_empirical = q_metrics["empirical_all_pair"]
    p_confident = p_metrics["confident_pair_direction"]
    q_confident = q_metrics["confident_pair_direction"]
    return {
        "candidate_count": len(candidates),
        "same_candidate_pair_support": True,
        "empirical_all_pair": {
            "non_tied_Q_pairs": p_empirical["non_tied_Q_pairs"],
            "scored_pairs": p_empirical["scored_pairs"],
            "p_accuracy": p_empirical["accuracy"],
            "q_accuracy": q_empirical["accuracy"],
            "q_minus_p_accuracy_difference": (
                q_empirical["accuracy"] - p_empirical["accuracy"]
                if p_empirical["accuracy"] is not None and q_empirical["accuracy"] is not None
                else None
            ),
        },
        "confident_pair_direction": {
            "eligible_pairs": p_confident["eligible_pairs"],
            "scored_pairs": p_confident["scored_pairs"],
            "p_accuracy": p_confident["accuracy"],
            "q_accuracy": q_confident["accuracy"],
            "q_minus_p_accuracy_difference": (
                q_confident["accuracy"] - p_confident["accuracy"]
                if p_confident["accuracy"] is not None and q_confident["accuracy"] is not None
                else None
            ),
        },
    }


def _candidate_summary(
    candidate: dict[str, Any], outcomes: list[dict[str, Any]], expected_k: int
) -> dict[str, Any]:
    result = {
        key: candidate[key]
        for key in (
            "instance_id",
            "sample",
            "p",
            "q",
            "solved",
            "proxy_valid",
            "invalid_body",
        )
    }
    result["outcome_replicates"] = len(outcomes)
    if len(outcomes) != expected_k:
        result.update(
            {
                "outcomes_complete": False,
                "oracle_successes": None,
                "Q": None,
                "wilson95": None,
            }
        )
        return result
    successes = sum(outcome["solved"] for outcome in outcomes)
    result.update(
        {
            "outcomes_complete": True,
            "oracle_successes": successes,
            "Q": successes / expected_k,
            "wilson95": wilson_interval(successes, expected_k),
        }
    )
    return result


def _summarize_task(
    task_id: str,
    candidates: list[dict[str, Any]],
    outcomes_by_candidate: dict[tuple[str, int], list[dict[str, Any]]],
    expected_candidates: int,
    expected_k: int,
) -> dict[str, Any]:
    candidates = sorted(
        candidates,
        key=lambda row: (0, row["sample"]) if isinstance(row["sample"], int) else (1, row["sample"]),
    )
    summarized = [
        _candidate_summary(
            candidate,
            outcomes_by_candidate.get((task_id, candidate["sample"]), []),
            expected_k,
        )
        for candidate in candidates
    ]
    complete = [row for row in summarized if row["outcomes_complete"]]
    incomplete_count = len(summarized) - len(complete)
    q_valid_candidates = sum(row["proxy_valid"] for row in summarized)
    q_valid_complete = [row for row in complete if row["q"] is not None]

    p_spearman = spearman_summary(
        [row["p"] for row in complete], [row["Q"] for row in complete]
    )
    q_spearman = spearman_summary(
        [row["q"] for row in q_valid_complete], [row["Q"] for row in q_valid_complete]
    )
    matched_p_spearman = spearman_summary(
        [row["p"] for row in q_valid_complete], [row["Q"] for row in q_valid_complete]
    )
    matched_q_spearman = spearman_summary(
        [row["q"] for row in q_valid_complete], [row["Q"] for row in q_valid_complete]
    )
    pairwise = {
        "p": _pairwise_metrics(complete, "p"),
        "q": _pairwise_metrics(complete, "q"),
    }

    def same_p_and_both_p1_unsolved(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return not left["solved"] and not right["solved"] and left["p"] == right["p"]

    main_subset_candidates = [row for row in complete if not row["solved"]]
    main_pair_universe = sum(
        left["p"] == right["p"]
        for left, right in combinations(main_subset_candidates, 2)
    )
    main_subset = {
        "rule": "both P1 candidates are unsolved and have exactly equal p",
        "unsolved_complete_candidate_count": len(main_subset_candidates),
        "same_p_candidate_pairs": main_pair_universe,
        "q_pairwise": _pairwise_metrics(
            complete, "q", pair_filter=same_p_and_both_p1_unsolved
        ),
    }

    return {
        "candidate_count": len(summarized),
        "expected_candidate_count": expected_candidates,
        "candidate_count_complete": len(summarized) == expected_candidates,
        "complete_outcome_candidates": len(complete),
        "incomplete_outcome_candidates": incomplete_count,
        "invalid_body_candidates": sum(row["invalid_body"] for row in summarized),
        "invalid_body_complete_outcome_candidates": sum(
            row["invalid_body"] for row in complete
        ),
        "outcome_rows_for_complete_candidates": len(complete) * expected_k,
        "task_complete": len(summarized) == expected_candidates and incomplete_count == 0,
        "proxy_coverage": {
            "candidate_count": len(summarized),
            "q_valid_candidates": q_valid_candidates,
            "q_missing_candidates": len(summarized) - q_valid_candidates,
            "q_valid_fraction": q_valid_candidates / len(summarized) if summarized else None,
            "complete_outcome_candidates": len(complete),
            "q_valid_complete_outcome_candidates": len(q_valid_complete),
            "q_valid_fraction_among_complete_outcomes": (
                len(q_valid_complete) / len(complete) if complete else None
            ),
        },
        "spearman": {"p_vs_Q": p_spearman, "q_vs_Q": q_spearman},
        "pairwise": pairwise,
        "matched_proxy_support": {
            "support_rule": "Both p and q use only candidates with valid q and all K terminal outcomes.",
            "candidate_count": len(q_valid_complete),
            "spearman": {
                "p_vs_Q": matched_p_spearman,
                "q_vs_Q": matched_q_spearman,
                "q_minus_p_rho_difference": (
                    matched_q_spearman["rho"] - matched_p_spearman["rho"]
                    if matched_p_spearman["rho"] is not None
                    and matched_q_spearman["rho"] is not None
                    else None
                ),
            },
            "pairwise": _matched_pairwise_metrics(q_valid_complete),
        },
        "main_subset": main_subset,
        "candidates": summarized,
    }


def _mean_task_metric(tasks: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float | None]) -> dict[str, Any]:
    values = [value for task in tasks if (value := getter(task)) is not None]
    return {
        "tasks_with_metric": len(values),
        "mean": math.fsum(values) / len(values) if values else None,
    }


def _paired_task_mean(
    tasks: list[dict[str, Any]],
    p_getter: Callable[[dict[str, Any]], float | None],
    q_getter: Callable[[dict[str, Any]], float | None],
) -> dict[str, Any]:
    """Average p, q, and q-minus-p over exactly the same eligible tasks."""
    pairs = [
        (p_value, q_value)
        for task in tasks
        if (p_value := p_getter(task)) is not None
        and (q_value := q_getter(task)) is not None
    ]
    count = len(pairs)
    p_mean = math.fsum(p_value for p_value, _ in pairs) / count if count else None
    q_mean = math.fsum(q_value for _, q_value in pairs) / count if count else None
    difference_mean = (
        math.fsum(q_value - p_value for p_value, q_value in pairs) / count
        if count
        else None
    )
    return {
        "tasks_with_both_metrics": count,
        "p_task_macro_mean": p_mean,
        "q_task_macro_mean": q_mean,
        "q_minus_p_task_macro_difference": difference_mean,
    }


def _task_macro(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(tasks),
        "spearman_p_vs_Q": _mean_task_metric(
            tasks, lambda task: task["spearman"]["p_vs_Q"]["rho"]
        ),
        "spearman_q_vs_Q": _mean_task_metric(
            tasks, lambda task: task["spearman"]["q_vs_Q"]["rho"]
        ),
        "p_empirical_all_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["pairwise"]["p"]["empirical_all_pair"]["accuracy"],
        ),
        "q_empirical_all_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["pairwise"]["q"]["empirical_all_pair"]["accuracy"],
        ),
        "p_confident_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["pairwise"]["p"]["confident_pair_direction"]["accuracy"],
        ),
        "q_confident_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["pairwise"]["q"]["confident_pair_direction"]["accuracy"],
        ),
        "main_subset_q_empirical_all_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["main_subset"]["q_pairwise"]["empirical_all_pair"]["accuracy"],
        ),
        "main_subset_q_empirical_pair_coverage": _mean_task_metric(
            tasks,
            lambda task: task["main_subset"]["q_pairwise"]["empirical_all_pair"]["coverage"],
        ),
        "main_subset_q_confident_pair_accuracy": _mean_task_metric(
            tasks,
            lambda task: task["main_subset"]["q_pairwise"]["confident_pair_direction"]["accuracy"],
        ),
        "main_subset_q_confident_pair_coverage": _mean_task_metric(
            tasks,
            lambda task: task["main_subset"]["q_pairwise"]["confident_pair_direction"][
                "coverage_among_confident_pairs"
            ],
        ),
        "q_candidate_coverage": _mean_task_metric(
            tasks, lambda task: task["proxy_coverage"]["q_valid_fraction"]
        ),
        "q_empirical_pair_coverage": _mean_task_metric(
            tasks, lambda task: task["pairwise"]["q"]["empirical_all_pair"]["coverage"]
        ),
        "q_confident_pair_coverage": _mean_task_metric(
            tasks,
            lambda task: task["pairwise"]["q"]["confident_pair_direction"][
                "coverage_among_confident_pairs"
            ],
        ),
        "matched_proxy_support": {
            "spearman_rho": _paired_task_mean(
                tasks,
                lambda task: task["matched_proxy_support"]["spearman"]["p_vs_Q"]["rho"],
                lambda task: task["matched_proxy_support"]["spearman"]["q_vs_Q"]["rho"],
            ),
            "empirical_pair_accuracy": _paired_task_mean(
                tasks,
                lambda task: task["matched_proxy_support"]["pairwise"]["empirical_all_pair"][
                    "p_accuracy"
                ],
                lambda task: task["matched_proxy_support"]["pairwise"]["empirical_all_pair"][
                    "q_accuracy"
                ],
            ),
            "confident_pair_accuracy": _paired_task_mean(
                tasks,
                lambda task: task["matched_proxy_support"]["pairwise"][
                    "confident_pair_direction"
                ]["p_accuracy"],
                lambda task: task["matched_proxy_support"]["pairwise"][
                    "confident_pair_direction"
                ]["q_accuracy"],
            ),
        },
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    run = _load_json(run_path)
    status = _required_string(run, "status", str(run_path))
    task_ids = run.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ValueError(f"{run_path}: task_ids must be a non-empty list of non-empty strings")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{run_path}: task_ids must be unique")
    expected_k = _required_int(run, "continuations_per_candidate", str(run_path), minimum=1)
    expected_candidates = _required_int(run, "candidates_per_task", str(run_path), minimum=1)

    candidate_path = run_dir / "candidates.jsonl"
    outcome_path = run_dir / "outcomes.jsonl"
    raw_candidates, candidates_file_exists = _load_jsonl(candidate_path)
    raw_outcomes, outcomes_file_exists = _load_jsonl(outcome_path)
    declared_tasks = set(task_ids)

    candidates_by_task: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    candidates_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_candidates:
        where = f"{candidate_path}:{row.pop('_line_number')}"
        instance_id = _required_string(row, "instance_id", where)
        if instance_id not in declared_tasks:
            raise ValueError(f"{where}: undeclared instance_id {instance_id!r}")
        sample = _required_int(row, "sample", where)
        if sample >= expected_candidates:
            raise ValueError(
                f"{where}: sample must be in [0, {expected_candidates})"
            )
        key = (instance_id, sample)
        if key in candidates_by_key:
            raise ValueError(f"{where}: duplicate candidate key {key!r}")
        row["p"] = _probability(row.get("p"), "p", where)
        row["q"] = _probability(row.get("q"), "q", where, nullable=True)
        row["solved"] = _required_bool(row, "solved", where)
        row["proxy_valid"] = _required_bool(row, "proxy_valid", where)
        row["invalid_body"] = _required_bool(row, "invalid_body", where)
        if row["proxy_valid"] != (row["q"] is not None):
            raise ValueError(f"{where}: proxy_valid must be true exactly when q is present")
        row["source_sha256"] = _required_string(row, "source_sha256", where)
        row["sample"] = sample
        candidates_by_key[key] = row
        candidates_by_task[instance_id].append(row)

    if any(len(rows) > expected_candidates for rows in candidates_by_task.values()):
        overflowing = {
            task_id: len(rows)
            for task_id, rows in candidates_by_task.items()
            if len(rows) > expected_candidates
        }
        raise ValueError(
            f"candidate counts exceed candidates_per_task={expected_candidates}: {overflowing}"
        )

    outcomes_by_candidate: dict[tuple[str, int], list[dict[str, Any]]] = {}
    outcome_keys: set[tuple[str, int, int]] = set()
    for row in raw_outcomes:
        where = f"{outcome_path}:{row.pop('_line_number')}"
        instance_id = _required_string(row, "instance_id", where)
        if instance_id not in declared_tasks:
            raise ValueError(f"{where}: undeclared instance_id {instance_id!r}")
        sample = _required_int(row, "sample", where)
        if sample >= expected_candidates:
            raise ValueError(
                f"{where}: sample must be in [0, {expected_candidates})"
            )
        replicate = _required_int(row, "replicate", where)
        if replicate >= expected_k:
            raise ValueError(f"{where}: replicate must be in [0, {expected_k})")
        key = (instance_id, sample)
        if key not in candidates_by_key:
            raise ValueError(f"{where}: outcome has no matching candidate key {key!r}")
        outcome_key = (instance_id, sample, replicate)
        if outcome_key in outcome_keys:
            raise ValueError(f"{where}: duplicate outcome key {outcome_key!r}")
        outcome_keys.add(outcome_key)
        row["solved"] = _required_bool(row, "solved", where)
        row["source_sha256"] = _required_string(row, "source_sha256", where)
        outcomes_by_candidate.setdefault(key, []).append(row)

    too_many_outcomes = {
        key: len(rows)
        for key, rows in outcomes_by_candidate.items()
        if len(rows) > expected_k
    }
    if too_many_outcomes:
        raise ValueError(
            f"outcome counts exceed continuations_per_candidate={expected_k}: {too_many_outcomes}"
        )

    tasks: dict[str, dict[str, Any]] = {}
    for task_id in sorted(task_ids):
        task_outcomes = {
            key: rows for key, rows in outcomes_by_candidate.items() if key[0] == task_id
        }
        tasks[task_id] = _summarize_task(
            task_id,
            candidates_by_task[task_id],
            task_outcomes,
            expected_candidates,
            expected_k,
        )

    complete_tasks = [task for task in tasks.values() if task["task_complete"]]
    observed_tasks = [task for task in tasks.values() if task["complete_outcome_candidates"] >= 2]
    status_complete = status.lower() == "complete"
    data_complete = all(task["task_complete"] for task in tasks.values())
    final_complete = status_complete and data_complete and candidates_file_exists and outcomes_file_exists
    distinguishable_pairs = sum(
        pair["eligible_pairs"]
        for task in tasks.values()
        for pair in [task["pairwise"]["p"]["confident_pair_direction"]]
    )
    if status.lower() == "failed":
        interpretation = "FAILED_NONFINAL"
    elif status.lower() == "running":
        interpretation = "PENDING"
    elif not status_complete:
        interpretation = "NONFINAL_STATUS"
    elif not data_complete or not candidates_file_exists or not outcomes_file_exists:
        interpretation = "INCOMPLETE"
    elif distinguishable_pairs == 0:
        interpretation = "INCONCLUSIVE"
    else:
        interpretation = "DESCRIPTIVE_ONLY"

    return {
        "schema_version": 1,
        "run_status": status,
        "interpretation_status": interpretation,
        "interpretation_rule": (
            "INCONCLUSIVE when a complete run has no within-task non-tied Q pair with disjoint "
            "two-sided 95% Wilson intervals; all reported associations are descriptive and no "
            "KEEP/KILL threshold is applied."
        ),
        "monte_carlo_target_note": (
            "Each empirical Q is successes/K from repeated frozen-policy continuations and is a "
            "noisy estimate, not ground truth. Wilson-filtered pairs are a conservative "
            "descriptive subset, without a multiple-testing claim."
        ),
        "completeness": {
            "candidate_file_present": candidates_file_exists,
            "outcome_file_present": outcomes_file_exists,
            "task_count_expected": len(task_ids),
            "task_count_with_full_candidate_panels": sum(
                task["candidate_count_complete"] for task in tasks.values()
            ),
            "task_count_with_all_candidate_outcomes": len(complete_tasks),
            "candidate_count_expected_per_task": expected_candidates,
            "continuations_expected_per_candidate": expected_k,
            "candidate_rows": len(raw_candidates),
            "outcome_rows": len(raw_outcomes),
            "run_status_complete": status_complete,
            "data_complete": data_complete,
            "complete": final_complete,
        },
        "distinguishable_Q_pair_count": distinguishable_pairs,
        "task_macro": {
            "weighting": "Equal weight per task; pairwise observations are not treated as independent significance tests.",
            "observed_tasks_with_at_least_two_complete_candidates": _task_macro(observed_tasks),
            "fully_complete_tasks_only": _task_macro(complete_tasks),
        },
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        summary = summarize_run(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output_path = args.output if args.output is not None else args.run_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
