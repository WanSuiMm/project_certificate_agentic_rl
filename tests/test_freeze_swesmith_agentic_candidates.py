import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_swesmith_agentic_candidates import changed_paths, eligible, select


class CandidateSelectionTest(unittest.TestCase):
    def row(self, image: str, index: int) -> dict:
        return {
            "instance_id": f"{image}.{index}", "image_name": image, "repo": image,
            "problem_statement": "Repair this issue.",
            "patch": "diff --git a/pkg/mod.py b/pkg/mod.py\n@@ -1 +1 @@\n-a\n+b\n",
            "FAIL_TO_PASS": ["test_a"], "PASS_TO_PASS": ["test_b"],
        }

    def test_single_production_file_only(self):
        self.assertEqual(changed_paths(self.row("img", 0)["patch"]), ["pkg/mod.py"])
        self.assertTrue(eligible(self.row("img", 0)))
        row = self.row("img", 0)
        row["patch"] += "diff --git a/pkg/other.py b/pkg/other.py\n"
        self.assertFalse(eligible(row))

    def test_balanced_repeatable_split(self):
        rows = [self.row(f"img{i}", j) for i in range(9) for j in range(10)]
        one, meta = select(rows)
        two, _ = select(list(reversed(rows)))
        self.assertEqual(one, two)
        self.assertEqual(len(one), 64)
        self.assertEqual(meta["split_counts"], {"train": 48, "heldout": 16})
        self.assertEqual(len({row["image_name"] for row in one}), 8)


if __name__ == "__main__":
    unittest.main()
