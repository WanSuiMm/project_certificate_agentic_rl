import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continue_swesmith_body_trajectories import feedback_from_public, load_census, public_fields


class TrajectoryProtocolTest(unittest.TestCase):
    def test_public_feedback_and_no_online_q(self):
        result = {"status": "scored", "public": {"valid": True, "passed": 3,
                  "total": 5, "exitcode": 1}, "p_T": 0.6, "solved": False}
        self.assertEqual(feedback_from_public(result),
                         "Public tests: 3/5 passed; exitcode=1.")
        self.assertNotIn("q", public_fields(result))
        with self.assertRaises(ValueError):
            public_fields({**result, "q": 0.8})

    def test_census_must_be_complete_and_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "run.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                load_census(path, {"ids": ["t"]})
            (path / "run.json").write_text(json.dumps({"status": "complete"}))
            (path / "results.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "28 x 16"):
                load_census(path, {"ids": ["t"]})


if __name__ == "__main__":
    unittest.main()
