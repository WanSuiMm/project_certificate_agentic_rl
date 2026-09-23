import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from swesmith_q_qualifier_worker import (enclosing_callable, make_bank,
                                         parameters, patch_location)


class QQualifierTest(unittest.TestCase):
    def test_diff_location_and_callable(self):
        patch = "diff --git a/pkg/mod.py b/pkg/mod.py\n@@ -2,3 +2,3 @@\n def f(s: str):\n-    return s.upper()\n+    return s.lower()\n"
        source = "x = 1\ndef f(s: str):\n    return s.upper()\n"
        path, positions = patch_location(patch)
        callable_path, node = enclosing_callable(source, positions)
        self.assertEqual(path, "pkg/mod.py")
        self.assertEqual(callable_path, ["f"])
        self.assertEqual(parameters(node, method=False), [("s", "str")])

    def test_bank_is_deterministic_and_unique(self):
        source = "def f(s: str):\n    return s.upper()\n"
        one = make_bank("task", source, [("s", "str")])
        two = make_bank("task", source, [("s", "str")])
        self.assertEqual(one, two)
        self.assertEqual(len(one), 256)
        self.assertEqual(len({tuple(x) for x in one}), 256)


if __name__ == "__main__":
    unittest.main()
