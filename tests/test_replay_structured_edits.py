from pathlib import Path
import json
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from replay_structured_edits import apply_call, confined_path, mutation_events


class ReplayStructuredEditsTest(unittest.TestCase):
    def test_failed_tool_attempt_is_not_a_successful_mutation(self) -> None:
        row = {
            "messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "bad",
                                "function": {
                                    "name": "str_replace_editor",
                                    "arguments": json.dumps(
                                        {
                                            "command": "str_replace",
                                            "path": "/testbed/a.py",
                                            "old_str": "missing",
                                            "new_str": "x",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_ids": ["bad"],
                        "content": [
                            {
                                "type": "text",
                                "text": "OBSERVATION:\nNo replacement was performed",
                            }
                        ],
                    },
                ]
            )
        }
        events = mutation_events(row)
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["succeeded"])

    def test_create_then_unique_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            created = apply_call(
                root,
                {
                    "command": "create",
                    "path": "/testbed/a.py",
                    "file_text": "value = 1\n",
                },
            )
            replaced = apply_call(
                root,
                {
                    "command": "str_replace",
                    "path": "/testbed/a.py",
                    "old_str": "value = 1",
                    "new_str": "value = 2",
                },
            )
            self.assertIsNotNone(created)
            self.assertIsNotNone(replaced)
            self.assertEqual((root / "a.py").read_text(), "value = 2\n")

    def test_rejects_escape_from_testbed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                confined_path(Path(directory), "/testbed/../../outside")

    def test_replace_fails_when_match_is_not_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("x x", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                apply_call(
                    root,
                    {
                        "command": "str_replace",
                        "path": "/testbed/a.py",
                        "old_str": "x",
                        "new_str": "y",
                    },
                )


if __name__ == "__main__":
    unittest.main()
