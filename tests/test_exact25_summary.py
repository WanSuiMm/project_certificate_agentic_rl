import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_exact25_intermediate import aggregate_summaries


def sample(valid=True, agrees=True):
    return {
        "traj_id": "repo.pr_1.example",
        "instance_id": "repo.pr_1",
        "declared_resolved": True,
        "observed_resolved": agrees,
        "endpoint_agrees": agrees,
        "initialization": {
            "git_apply_check_returncode": 0,
            "git_apply_returncode": 0,
        },
        "state_count": 2,
        "mutation_count": 1,
        "curve": [
            {"state_index": 0, "F": 1 if valid else None, "R": 0 if valid else None, "valid": valid},
            {"state_index": 1, "F": 0 if valid else None, "R": 0 if valid else None, "valid": valid},
        ],
        "transitions": [{"from": 0, "to": 1, "label": "positive" if valid else "invalid"}],
    }


class Exact25SummaryTests(unittest.TestCase):
    def test_valid_endpoint_passes(self):
        result = aggregate_summaries([sample()], expected_tasks=1)
        self.assertTrue(result["all_endpoint_gates_pass"])
        self.assertEqual(result["endpoint_gate_pass_count"], 1)
        self.assertEqual(result["transition_counts"], {"positive": 1})

    def test_invalid_endpoint_does_not_pass_despite_agreement(self):
        result = aggregate_summaries([sample(valid=False)], expected_tasks=1)
        self.assertEqual(result["endpoint_agreement_count"], 1)
        self.assertEqual(result["endpoint_gate_pass_count"], 0)
        self.assertEqual(result["valid_state_count"], 0)

    def test_incomplete_batch_does_not_pass(self):
        result = aggregate_summaries([sample()], expected_tasks=25)
        self.assertFalse(result["complete"])
        self.assertFalse(result["all_endpoint_gates_pass"])


if __name__ == "__main__":
    unittest.main()
