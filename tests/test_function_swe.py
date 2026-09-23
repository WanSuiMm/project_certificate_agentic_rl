import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from function_swe import (
    TaskError, load_task, observe, public_prompt, replace_top_level_function,
    score_source, sha256_text,
)
from function_swe_executor import CommandExecutor


def toy_task():
    source = "def double(x):\n    return x + 1\n"
    return {
        "task_id": "toy.double", "issue": "Double the input", "target_function": "double",
        "buggy_source": source, "buggy_sha256": sha256_text(source),
        "reference_sha256": "a" * 64,
        "public_cases": [{"args": [2], "kwargs": {}, "expect": {"kind": "return", "value": 4}}],
        "hidden_cases": [{"args": [3], "kwargs": {}, "expect": {"kind": "return", "value": 6}}],
        "probe_cases": [{"args": [4], "kwargs": {}, "expect": {"kind": "return", "value": 8}}],
    }


class FunctionSweTests(unittest.TestCase):
    def test_function_replacement_is_single_target(self):
        source = "import math\n\ndef double(x):\n    return x + 1\n\ndef other():\n    return 7\n"
        edited = replace_top_level_function(source, "double", "def double(x):\n    return x * 2")
        self.assertIn("return x * 2", edited)
        self.assertIn("def other():", edited)
        with self.assertRaises(TaskError):
            replace_top_level_function(source, "double", "import os\ndef double(x): return x * 2")

    def test_score_public_and_terminal(self):
        task = toy_task()
        initial = score_source(task, task["buggy_source"], terminal=True)
        self.assertEqual((initial["p_T"], initial["q"], initial["solved"]), (0.0, 0.0, False))
        fixed = replace_top_level_function(task["buggy_source"], "double", "def double(x): return x * 2")
        terminal = score_source(task, fixed, terminal=True)
        self.assertEqual((terminal["p_T"], terminal["q"], terminal["solved"]), (1.0, 1.0, True))
        public = score_source(task, fixed, terminal=False)
        self.assertNotIn("q", public)
        self.assertNotIn("solved", public)
        eval_terminal = score_source(task, fixed, terminal=True, include_proxy=False)
        self.assertTrue(eval_terminal["solved"])
        self.assertNotIn("q", eval_terminal)

    def test_prompt_does_not_contain_hidden_or_probe_values(self):
        task = toy_task()
        prompt = public_prompt(task, task["buggy_source"], "0/1")
        self.assertIn("Double the input", prompt)
        self.assertNotIn('"value":6', prompt)
        self.assertNotIn('"value":8', prompt)

    def test_worker_protocol_on_trusted_fixture(self):
        task = toy_task()
        command = [sys.executable, str(ROOT / "scripts" / "function_swe_worker.py")]
        result = CommandExecutor(command).score(task, task["buggy_source"], terminal=True)
        self.assertFalse(result["solved"])

    def test_observation_keeps_tuple_and_exception_type(self):
        self.assertEqual(observe(lambda x: (x, True), {"args": [3], "kwargs": {}}),
                         {"kind": "return", "value": {"__tuple__": [3, True]}})
        self.assertEqual(observe(lambda x: 1 / x, {"args": [0], "kwargs": {}}),
                         {"kind": "exception", "type": "ZeroDivisionError"})


if __name__ == "__main__":
    unittest.main()
