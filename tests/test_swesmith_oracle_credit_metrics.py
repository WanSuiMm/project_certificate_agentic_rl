import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_swesmith_oracle_credit import summarize_run


def candidate(sample, *, p, q, solved=False, proxy_valid=True, invalid_body=False):
    return {
        "instance_id": "task-a",
        "sample": sample,
        "source_sha256": f"p1-{sample}",
        "p": p,
        "q": q,
        "solved": solved,
        "proxy_valid": proxy_valid,
        "invalid_body": invalid_body,
    }


def outcome(sample, replicate, solved):
    return {
        "instance_id": "task-a",
        "sample": sample,
        "replicate": replicate,
        "solved": solved,
        "source_sha256": f"final-{sample}-{replicate}",
    }


def write_run(root: Path, candidates, outcomes, *, status="complete", k=4):
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": status,
                "task_ids": ["task-a"],
                "continuations_per_candidate": k,
                "candidates_per_task": len(candidates),
            }
        ),
        encoding="utf-8",
    )
    (root / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    (root / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8"
    )


class SweSmithOracleCreditMetricsTest(unittest.TestCase):
    def test_all_zero_oracle_q_is_inconclusive_and_correlations_are_null(self):
        rows = [candidate(i, p=0.1 * i, q=0.2 * i) for i in range(3)]
        outcomes = [outcome(row["sample"], replicate, False) for row in rows for replicate in range(4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            summary = summarize_run(root)

        task = summary["tasks"]["task-a"]
        self.assertTrue(summary["completeness"]["complete"])
        self.assertEqual(summary["interpretation_status"], "INCONCLUSIVE")
        self.assertEqual(task["spearman"]["p_vs_Q"]["rho"], None)
        self.assertEqual(task["spearman"]["p_vs_Q"]["null_reason"], "constant_Q")
        self.assertEqual(task["spearman"]["q_vs_Q"]["rho"], None)
        self.assertIsNone(task["pairwise"]["p"]["empirical_all_pair"]["accuracy"])
        self.assertEqual(summary["distinguishable_Q_pair_count"], 0)

    def test_tied_proxy_scores_receive_half_credit(self):
        rows = [candidate(0, p=0.5, q=0.7), candidate(1, p=0.5, q=0.7)]
        outcomes = [
            *(outcome(0, replicate, False) for replicate in range(4)),
            *(outcome(1, replicate, True) for replicate in range(4)),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            task = summarize_run(root)["tasks"]["task-a"]

        for proxy in ("p", "q"):
            self.assertEqual(task["pairwise"][proxy]["empirical_all_pair"]["accuracy"], 0.5)
            self.assertEqual(task["pairwise"][proxy]["confident_pair_direction"]["accuracy"], 0.5)
        self.assertEqual(
            task["pairwise"]["q"]["confident_pair_direction"]["eligible_pairs"], 1
        )

    def test_known_ordering_has_perfect_spearman_and_pairwise_accuracy(self):
        rates = [0, 1, 3, 4]
        rows = [
            candidate(i, p=(i + 1) / 5, q=(i + 1) / 5)
            for i in range(len(rates))
        ]
        outcomes = [
            outcome(row["sample"], replicate, replicate < rates[row["sample"]])
            for row in rows
            for replicate in range(4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            task = summarize_run(root)["tasks"]["task-a"]

        self.assertAlmostEqual(task["spearman"]["p_vs_Q"]["rho"], 1.0)
        self.assertAlmostEqual(task["spearman"]["q_vs_Q"]["rho"], 1.0)
        self.assertEqual(task["pairwise"]["p"]["empirical_all_pair"]["accuracy"], 1.0)
        self.assertEqual(task["pairwise"]["q"]["empirical_all_pair"]["accuracy"], 1.0)

    def test_null_q_is_missing_coverage_and_not_a_zero_score(self):
        rows = [
            candidate(0, p=0.1, q=0.2),
            candidate(1, p=0.2, q=None, proxy_valid=False),
            candidate(2, p=0.3, q=0.8),
        ]
        rates = [0, 2, 4]
        outcomes = [
            outcome(row["sample"], replicate, replicate < rates[row["sample"]])
            for row in rows
            for replicate in range(4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            task = summarize_run(root)["tasks"]["task-a"]

        self.assertEqual(task["proxy_coverage"]["q_missing_candidates"], 1)
        self.assertAlmostEqual(task["proxy_coverage"]["q_valid_fraction"], 2 / 3)
        q_pairs = task["pairwise"]["q"]["empirical_all_pair"]
        self.assertEqual(q_pairs["non_tied_Q_pairs"], 3)
        self.assertEqual(q_pairs["scored_pairs"], 1)
        self.assertAlmostEqual(q_pairs["coverage"], 1 / 3)
        self.assertEqual(task["candidates"][1]["q"], None)

    def test_duplicate_and_incomplete_outcomes_are_rejected_or_left_unscored(self):
        rows = [candidate(0, p=0.5, q=0.4)]
        duplicate = [outcome(0, 0, False), outcome(0, 0, True)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate outcome key"):
                summarize_run(root)

        partial = [outcome(0, replicate, False) for replicate in range(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, partial, status="running")
            summary = summarize_run(root)

        task = summary["tasks"]["task-a"]
        self.assertFalse(summary["completeness"]["complete"])
        self.assertEqual(summary["interpretation_status"], "PENDING")
        self.assertEqual(task["incomplete_outcome_candidates"], 1)
        self.assertIsNone(task["candidates"][0]["Q"])
        self.assertEqual(task["pairwise"]["p"]["empirical_all_pair"]["non_tied_Q_pairs"], 0)

    def test_sample_and_replicate_indices_must_be_within_declared_ranges(self):
        rows = [candidate(0, p=0.5, q=0.4)]
        out_of_range_replicate = [outcome(0, replicate, False) for replicate in (0, 1, 2, 4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, out_of_range_replicate)
            with self.assertRaisesRegex(ValueError, r"replicate must be in \[0, 4\)"):
                summarize_run(root)

        out_of_range_sample = [candidate(1, p=0.5, q=0.4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, out_of_range_sample, [])
            with self.assertRaisesRegex(ValueError, r"sample must be in \[0, 1\)"):
                summarize_run(root)

    def test_matched_proxy_metrics_avoid_missing_q_selection_bias(self):
        rates = [0, 1, 2, 3, 4]
        p_scores = [0.8, 0.6, 0.4, 0.9, 1.0]
        q_scores = [0.9, 0.5, 0.1, None, None]
        rows = [
            candidate(
                i,
                p=p_scores[i],
                q=q_scores[i],
                proxy_valid=q_scores[i] is not None,
            )
            for i in range(len(rates))
        ]
        outcomes = [
            outcome(row["sample"], replicate, replicate < rates[row["sample"]])
            for row in rows
            for replicate in range(4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            summary = summarize_run(root)

        task = summary["tasks"]["task-a"]
        matched = task["matched_proxy_support"]
        self.assertAlmostEqual(task["spearman"]["p_vs_Q"]["rho"], 0.6)
        self.assertEqual(matched["candidate_count"], 3)
        self.assertAlmostEqual(matched["spearman"]["p_vs_Q"]["rho"], -1.0)
        self.assertAlmostEqual(matched["spearman"]["q_vs_Q"]["rho"], -1.0)
        self.assertAlmostEqual(matched["spearman"]["q_minus_p_rho_difference"], 0.0)
        self.assertEqual(matched["pairwise"]["empirical_all_pair"]["non_tied_Q_pairs"], 3)
        self.assertEqual(matched["pairwise"]["empirical_all_pair"]["scored_pairs"], 3)
        self.assertEqual(matched["pairwise"]["empirical_all_pair"]["p_accuracy"], 0.0)
        self.assertEqual(matched["pairwise"]["empirical_all_pair"]["q_accuracy"], 0.0)
        self.assertAlmostEqual(
            matched["pairwise"]["empirical_all_pair"]["q_minus_p_accuracy_difference"],
            0.0,
        )
        macro = summary["task_macro"]["fully_complete_tasks_only"]["matched_proxy_support"]
        self.assertEqual(macro["spearman_rho"]["tasks_with_both_metrics"], 1)
        self.assertEqual(macro["spearman_rho"]["p_task_macro_mean"], -1.0)
        self.assertEqual(macro["spearman_rho"]["q_task_macro_mean"], -1.0)
        self.assertEqual(macro["spearman_rho"]["q_minus_p_task_macro_difference"], 0.0)

    def test_main_subset_requires_unsolved_candidates_with_exactly_tied_p(self):
        rows = [
            candidate(0, p=0.5, q=0.1, solved=False),
            candidate(1, p=0.5, q=0.9, solved=False),
            candidate(2, p=0.7, q=0.8, solved=False),
            candidate(3, p=0.5, q=0.95, solved=True),
        ]
        rates = [0, 4, 2, 1]
        outcomes = [
            outcome(row["sample"], replicate, replicate < rates[row["sample"]])
            for row in rows
            for replicate in range(4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, rows, outcomes)
            main = summarize_run(root)["tasks"]["task-a"]["main_subset"]

        self.assertEqual(main["unsolved_complete_candidate_count"], 3)
        self.assertEqual(main["same_p_candidate_pairs"], 1)
        self.assertEqual(
            main["q_pairwise"]["empirical_all_pair"]["non_tied_Q_pairs"], 1
        )
        self.assertEqual(main["q_pairwise"]["empirical_all_pair"]["accuracy"], 1.0)

    def test_candidate_flags_and_probability_bounds_are_validated(self):
        bad_rows = [candidate(0, p=1.01, q=0.2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, bad_rows, [])
            with self.assertRaisesRegex(ValueError, "p must be"):
                summarize_run(root)


if __name__ == "__main__":
    unittest.main()
