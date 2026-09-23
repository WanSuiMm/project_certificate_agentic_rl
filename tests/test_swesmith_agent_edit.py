import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from swesmith_agent_edit import InvalidEdit, replace_callable


class AgentEditTest(unittest.TestCase):
    def test_method(self):
        source = "class A:\n    def f(self, x: int):\n        return x\n\n    def g(self):\n        return 2\n"
        edited = replace_callable(source, ["A", "f"], "def f(self, x: int):\n    return x + 1")
        self.assertIn("    def f(self, x: int):\n        return x + 1", edited)
        self.assertIn("    def g(self):", edited)

    def test_signature_rejected(self):
        with self.assertRaises(InvalidEdit):
            replace_callable("def f(x):\n    return x\n", ["f"], "def f(y):\n    return y")

    def test_top_level(self):
        self.assertEqual(replace_callable("def f(x):\n    return x\n", ["f"],
                                          "def f(x):\n    return x + 1"),
                         "def f(x):\n    return x + 1\n")
