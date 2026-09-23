import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_function_task import freeze_spec
from function_swe import TaskError


def spec():
    return {
        "task_id": "toy.double", "issue": "Double x", "target_function": "double",
        "buggy_source": "def double(x):\n    return x + 1\n",
        "reference_source": "def double(x):\n    return x * 2\n",
        "public_cases": [{"args": [2], "kwargs": {}, "expect": {"kind": "return", "value": 4}}],
        "hidden_cases": [{"args": [3], "kwargs": {}, "expect": {"kind": "return", "value": 6}}],
        "probe_inputs": [{"args": [4], "kwargs": {}}],
        "provenance": {"source": "trusted-test-only"},
    }


class FreezeTaskTests(unittest.TestCase):
    def test_freezes_reference_probe_without_exposing_reference_source(self):
        result = freeze_spec(spec(), expected_probes=1)
        self.assertEqual(result["probe_cases"][0]["expect"], {"kind": "return", "value": 8})
        self.assertNotIn("reference_source", result)

    def test_rejects_probe_overlap(self):
        candidate = spec()
        candidate["probe_inputs"] = [{"args": [2], "kwargs": {}}]
        with self.assertRaises(TaskError):
            freeze_spec(candidate, expected_probes=1)

    def test_rejects_wrong_reference(self):
        candidate = spec()
        candidate["reference_source"] = "def double(x):\n    return x + 1\n"
        with self.assertRaises(TaskError):
            freeze_spec(candidate, expected_probes=1)


if __name__ == "__main__":
    unittest.main()
