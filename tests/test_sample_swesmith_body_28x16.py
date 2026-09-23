import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sample_swesmith_body_28x16 import information_metrics


class CensusMetricsTest(unittest.TestCase):
    def test_q_splits_test_tie_only_among_executable_candidates(self):
        rows = []
        for sample in range(16):
            p, q = (0.5, 0.25) if sample < 8 else (0.5, 0.75)
            rows.append({"instance_id": "t", "sample": sample, "status": "scored",
                         "invalid_edits": 0, "score": {"p_T": p, "q": q}})
        result = information_metrics(rows, ["t"])
        self.assertEqual(result["complete_tasks"], 1)
        self.assertEqual(result["syntactically_valid_action_rate"], 1)
        self.assertEqual(result["executable_candidate_rate"], 1)
        self.assertEqual(result["test_informative_tasks_executable"], 0)
        self.assertEqual(result["q_informative_tasks_executable"], 1)
        self.assertEqual(result["same_test_pairs_executable"], 120)
        self.assertEqual(result["q_splits_test_tie_pairs_executable"], 64)

    def test_incomplete_task_is_not_counted_as_evidence(self):
        rows = [{"instance_id": "t", "sample": 0, "status": "task_error"}]
        result = information_metrics(rows, ["t"])
        self.assertEqual(result["complete_tasks"], 0)
        self.assertEqual(result["q_informative_tasks_all"], 0)


if __name__ == "__main__":
    unittest.main()
