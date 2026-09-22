import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_swesmith_patch_alignment import audit_row


def call(path: str, command: str = "str_replace") -> dict:
    return {
        "function": {
            "name": "str_replace_editor",
            "arguments": json.dumps({"command": command, "path": path}),
        }
    }


class PatchAlignmentTest(unittest.TestCase):
    def test_detects_aligned_path(self) -> None:
        row = {
            "traj_id": "t",
            "instance_id": "i",
            "resolved": True,
            "messages": [{"tool_calls": [call("/testbed/pkg/a.py")]}],
            "patch": "diff --git a/pkg/a.py b/pkg/a.py\n",
        }
        self.assertEqual(audit_row(row)["classification"], "aligned_non_create_path")

    def test_detects_disjoint_path(self) -> None:
        row = {
            "traj_id": "t",
            "instance_id": "i",
            "resolved": True,
            "messages": [{"tool_calls": [call("/testbed/pkg/a.py")]}],
            "patch": "diff --git a/other/b.py b/other/b.py\n",
        }
        self.assertEqual(audit_row(row)["classification"], "disjoint_nonempty_patch")


if __name__ == "__main__":
    unittest.main()
