import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from swesmith_agent_edit import InvalidEdit, current_callable_body, replace_callable_body


class AgentBodyEditTest(unittest.TestCase):
    def test_method_preserves_decorator_name_and_signature(self):
        source = "class A:\n    @staticmethod\n    def f(x: int) -> int:\n        return x\n\n    def g(self):\n        return 2\n"
        edited = replace_callable_body(source, ["A", "f"], "return x + 1")
        self.assertIn("    @staticmethod\n    def f(x: int) -> int:\n        return x + 1", edited)
        self.assertIn("    def g(self):\n        return 2", edited)

    def test_fence_and_body_import(self):
        source = "def f(x):\n    return x\n"
        edited = replace_callable_body(source, ["f"], "```python\nimport math\nreturn math.floor(x)\n```")
        self.assertEqual(edited, "def f(x):\n    import math\n    return math.floor(x)\n")

    def test_fenced_indented_body_keeps_relative_indent(self):
        source = "def f(x):\n    return x\n"
        edited = replace_callable_body(source, ["f"],
                                       "```python\n        if x:\n            return 1\n        return 0\n```")
        self.assertEqual(edited, "def f(x):\n    if x:\n        return 1\n    return 0\n")

    def test_invalid_body(self):
        with self.assertRaises(InvalidEdit):
            replace_callable_body("def f(x):\n    return x\n", ["f"], "if x:\nreturn 1")

    def test_inline_body(self):
        self.assertEqual(replace_callable_body("def f(x): return x\n", ["f"], "return x + 1"),
                         "def f(x):\n    return x + 1\n")

    def test_saved_body_matches_assembled_state(self):
        source = "def f(x):\n    return x\n"
        edited = replace_callable_body(source, ["f"],
                                       "```python\n        if x:\n            return 1\n        return 0\n```")
        self.assertEqual(current_callable_body(edited, ["f"]),
                         "if x:\n    return 1\nreturn 0")
        self.assertEqual(replace_callable_body(source, ["f"],
                                               current_callable_body(edited, ["f"])), edited)


if __name__ == "__main__":
    unittest.main()
