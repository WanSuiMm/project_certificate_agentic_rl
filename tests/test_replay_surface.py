import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_replay_surface import audit_trajectory


def tool(name: str, arguments: dict) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


class ReplaySurfaceTest(unittest.TestCase):
    def test_counts_undo_and_shell_mutation(self) -> None:
        row = {
            "traj_id": "t",
            "instance_id": "i",
            "resolved": False,
            "messages": [
                {
                    "tool_calls": [
                        tool(
                            "str_replace_editor",
                            {"command": "undo_edit", "path": "/testbed/a.py"},
                        ),
                        tool("bash", {"command": "sed -i 's/a/b/' a.py"}),
                    ]
                }
            ],
        }
        result = audit_trajectory(row)
        self.assertEqual(result["undo_edit_count"], 1)
        self.assertEqual(result["suspicious_shell_mutation_count"], 1)
        self.assertIn(
            "sed_in_place", result["suspicious_shell_mutations"][0]["patterns"]
        )

    def test_ignores_read_only_shell(self) -> None:
        row = {
            "traj_id": "t",
            "instance_id": "i",
            "resolved": True,
            "messages": [{"tool_calls": [tool("bash", {"command": "grep -R x ."})]}],
        }
        result = audit_trajectory(row)
        self.assertEqual(result["shell_call_count"], 1)
        self.assertEqual(result["suspicious_shell_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
