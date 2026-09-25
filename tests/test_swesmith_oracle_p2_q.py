import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_swesmith_oracle_p2_q import summarize


class OracleP2QTest(unittest.TestCase):
    def setUp(self):
        self.oracle = ROOT / "evidence" / "oracle_credit_6x16x4_v01"
        self.p2 = ROOT / "evidence" / "oracle_credit_p2_q_v01" / "observations.jsonl"

    def test_frozen_source_free_result(self):
        result = summarize(self.oracle, self.p2)
        self.assertEqual(result["first_hit_trajectories"], 53)
        self.assertEqual(result["first_hit_by_step"]["2"], 13)
        self.assertEqual(result["P2_q_valid_observations"], 373)
        self.assertEqual(result["fully_matched_candidates"], 87)
        self.assertEqual(result["informative_tasks"], 3)

    def test_rejects_misaligned_p2_source(self):
        lines = self.p2.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["source_sha256"] = "0" * 64
        lines[0] = json.dumps(row)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not aligned"):
                summarize(self.oracle, path)


if __name__ == "__main__":
    unittest.main()
