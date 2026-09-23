import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continue_swesmith_body_trajectories import feedback_from_public, ready_task_census, public_fields
from capture_swesmith_body_trajectories import (
    pilot_ids, public_prompt_item, run_step, validate_feedback_chain,
)
from grade_swesmith_trajectory_q import validate_capture_receipt


class TrajectoryProtocolTest(unittest.TestCase):
    def test_public_feedback_and_no_online_q(self):
        result = {"status": "scored", "public": {"valid": True, "passed": 3,
                  "total": 5, "exitcode": 1}, "p_T": 0.6, "solved": False}
        self.assertEqual(feedback_from_public(result),
                         "Public tests: 3/5 passed; exitcode=1.")
        self.assertNotIn("q", public_fields(result))
        with self.assertRaises(ValueError):
            public_fields({**result, "q": 0.8})

    def test_census_ready_block_is_public_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "results.jsonl").write_text("", encoding="utf-8")
            self.assertIsNone(ready_task_census(path, "t", 0))
            rows = [{"instance_id": "t", "sample": i, "status": "scored",
                     "completion": "return x", "final_source_sha256": "abc",
                     "generated_tokens": 3,
                     "score": {"public": {"passed": 0, "total": 1}, "p_T": 0,
                               "solved": False, "status": "scored", "q": 0.9}}
                    for i in range(16)]
            (path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            block = ready_task_census(path, "t", 0)
            self.assertEqual(len(block), 16)
            self.assertNotIn("q", block[0]["score"])
            self.assertIsNone(ready_task_census(path, "next", 1))

    def test_fresh_closed_loop_step_uses_public_feedback_and_never_q(self):
        class PublicSession:
            def call(self, **kwargs):
                self_args = kwargs
                self_test.assertEqual(self_args["mode"], "score")
                self_test.assertFalse(self_args["include_proxy"])
                self_test.assertNotIn("q_bank", self_args)
                return {"status": "scored", "public": {"valid": True,
                        "passed": 2, "total": 3, "exitcode": 1},
                        "p_T": 2 / 3, "solved": False}

        self_test = self
        source = "def f(x):\n    return x\n"
        prior = {"source": source, "public": {"valid": True, "passed": 1,
                 "total": 3, "exitcode": 1}, "p_T": 1 / 3, "solved": False}
        item = {"task": {"problem_statement": "repair f"},
                "q_bank": {"module": "m", "callable": ["f"],
                           "reference": ["SECRET_Q_BANK_SENTINEL"]}}
        action = SimpleNamespace(completion="return x + 1", token_ids=[1, 2, 3],
                                 prompt_length=1)
        with patch("capture_swesmith_body_trajectories.generate_action", return_value=action) as generate:
            result = run_step(item=item, prior=prior, path=["f"],
                              model=None, tokenizer=None, session=PublicSession(),
                              config={"context_tokens": 8192,
                                      "max_generated_tokens_per_edit": 1024},
                              device="cpu", seed=0)
        prompt = generate.call_args.args[2]
        self.assertIn("Public tests: 1/3 passed", prompt)
        self.assertNotIn("SECRET_Q_BANK_SENTINEL", prompt)
        self.assertIn("return x + 1", result["source"])
        self.assertEqual(result["p_T"], 2 / 3)
        self.assertNotIn("q", result)

    def test_invalid_edit_is_reported_in_next_feedback(self):
        score = {"invalid_body": True, "public": {"valid": True,
                 "passed": 1, "total": 3, "exitcode": 1}}
        feedback = feedback_from_public(score)
        self.assertIn("invalid; source unchanged", feedback)
        self.assertIn("Public tests: 1/3", feedback)

    def test_pilot_subset_and_offline_receipt_reject_open_loop(self):
        selection = {"status": "q_first_frozen", "count": 28,
                     "ids": [f"task-{i}" for i in range(28)]}
        config = {"pilot_tasks": 8, "trajectories_per_task": 16, "steps": 8}
        ids = pilot_ids(selection, config)
        self.assertEqual(len(ids), 8)
        self.assertEqual(public_prompt_item({"task": {}, "q_bank": {
            "module": "m", "callable": ["f"], "reference": [1]}})["q_bank"],
            {"module": "m", "callable": ["f"]})
        receipt = {"kind": "fresh_eight_step_closed_loop_public_feedback_q_offline",
                   "status": "complete", "task_ids": ids, "tasks": 8,
                   "trajectories_per_task": 16, "steps": 8,
                   "selection_sha256": "selection", "states": 1032}
        self.assertEqual(validate_capture_receipt(receipt, selection, "selection")[-1], 1032)
        with self.assertRaises(RuntimeError):
            validate_capture_receipt({**receipt, "kind": "eight_step_capture_only_no_online_tests_or_q"},
                                     selection, "selection")

    def test_saved_feedback_chain_rejects_open_loop_or_changed_invalid_source(self):
        public = {"valid": True, "passed": 1, "total": 3, "exitcode": 1}
        p0 = {"source_sha256": "source0", "public": public,
              "feedback_given": None}
        p1 = {"source_sha256": "source0", "public": public,
              "invalid_body": True, "feedback_given": feedback_from_public(p0)}
        states = {("task", None, 0): p0, ("task", 0, 1): p1}
        validate_feedback_chain(states)
        with self.assertRaisesRegex(RuntimeError, "broken public-feedback chain"):
            validate_feedback_chain({**states, ("task", 0, 1): {
                **p1, "feedback_given": "No tests were run"}})
        with self.assertRaisesRegex(RuntimeError, "invalid edit changed source"):
            validate_feedback_chain({**states, ("task", 0, 1): {
                **p1, "source_sha256": "different"}})


if __name__ == "__main__":
    unittest.main()
