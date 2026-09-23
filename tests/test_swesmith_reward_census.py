import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_swesmith_reward_census import summarize


class RewardCensusTest(unittest.TestCase):
    def test_semantic_reward_splits_unresolved_test_tie(self):
        rows = []
        for sample in range(16):
            q = sample / 16
            rows.append({"instance_id": "task", "sample": sample, "status": "scored",
                         "score": {"solved": False}, "test_reward": 0.0,
                         "semantic_reward": 0.5 * q})
        summary = summarize(rows, ["task"])
        self.assertEqual(summary["tasks_with_test_reward_variance"], 0)
        self.assertEqual(summary["tasks_with_semantic_reward_variance"], 1)
        self.assertEqual(summary["unresolved_test_tie_pairs"], 120)
        self.assertEqual(summary["unresolved_ties_split_by_semantic"], 120)


if __name__ == "__main__":
    unittest.main()
