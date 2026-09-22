import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from select_swesmith_tool_trajectories import extract_structured_edits, select_panel


def row(instance: str, resolved: bool, edit_count: int) -> dict:
    messages = []
    for index in range(edit_count):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": json.dumps(
                                {
                                    "command": "str_replace",
                                    "path": f"/testbed/file{index}.py",
                                }
                            ),
                        },
                    }
                ],
            }
        )
    return {
        "messages": json.dumps(messages),
        "instance_id": instance,
        "resolved": resolved,
        "model": "test-model",
        "traj_id": f"{instance}.traj",
        "patch": "",
    }


class SweSmithSelectionTest(unittest.TestCase):
    def test_extracts_only_explicit_editor_mutations(self) -> None:
        messages = [
            {
                "tool_calls": [
                    {
                        "id": "view",
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": '{"command":"view","path":"a.py"}',
                        },
                    },
                    {
                        "id": "edit",
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": '{"command":"insert","path":"a.py"}',
                        },
                    },
                    {
                        "id": "shell",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"sed -i s/a/b/ a.py"}',
                        },
                    },
                ]
            }
        ]
        edits = extract_structured_edits(messages)
        self.assertEqual([edit["tool_call_id"] for edit in edits], ["edit"])

    def test_balances_outcomes_and_repositories(self) -> None:
        rows = []
        for repo_index in range(4):
            rows.append(row(f"repo{repo_index}.task-a", True, 2))
            rows.append(row(f"repo{repo_index}.task-b", False, 2))
        selected, summary = select_panel(
            rows,
            target_per_outcome=4,
            min_edits=2,
            max_per_repo=2,
            min_repos=4,
            max_scanned=100,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(summary["resolved"], 4)
        self.assertEqual(summary["unresolved"], 4)
        self.assertEqual(summary["repository_count"], 4)


if __name__ == "__main__":
    unittest.main()
