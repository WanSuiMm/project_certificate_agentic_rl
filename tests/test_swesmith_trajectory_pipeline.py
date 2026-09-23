import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continue_swesmith_body_trajectories import feedback_from_public, ready_task_census, public_fields


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


if __name__ == "__main__":
    unittest.main()
