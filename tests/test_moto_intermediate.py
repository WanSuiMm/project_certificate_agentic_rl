from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_moto_intermediate import parse_junit, transition_label


class MotoIntermediateTest(unittest.TestCase):
    def test_transition_labels(self) -> None:
        def state(f: int, r: int, valid: bool = True) -> dict:
            return {"F": f, "R": r, "valid": valid}

        self.assertEqual(transition_label(state(2, 0), state(1, 0)), "positive")
        self.assertEqual(transition_label(state(1, 0), state(2, 0)), "negative")
        self.assertEqual(transition_label(state(2, 0), state(1, 1)), "mixed")
        self.assertEqual(transition_label(state(2, 0), state(2, 0)), "neutral")
        self.assertEqual(transition_label(state(2, 0, False), state(1, 0)), "invalid")

    def test_junit_parser(self) -> None:
        xml = """<testsuite tests="3">
        <testcase name="a" />
        <testcase name="b"><failure /></testcase>
        <testcase name="c"><error /></testcase>
        </testsuite>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xml"
            path.write_text(xml, encoding="utf-8")
            result = parse_junit(path, 3)
        self.assertTrue(result["valid"])
        self.assertEqual(result["nonpassing"], 2)


if __name__ == "__main__":
    unittest.main()
