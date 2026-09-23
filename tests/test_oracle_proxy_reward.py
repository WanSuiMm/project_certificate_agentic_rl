import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from oracle_proxy_reward import terminal_reward


class OracleProxyRewardTests(unittest.TestCase):
    def test_only_terminal_metric_enters_each_arm(self):
        self.assertEqual(terminal_reward("terminal", solved=False), 0.0)
        self.assertEqual(terminal_reward("terminal", solved=True), 1.0)
        self.assertEqual(terminal_reward("test", solved=False, public_pass_fraction=0.4), 0.2)
        self.assertEqual(terminal_reward("semantic", solved=False, reference_agreement=0.4), 0.2)

    def test_actual_proxy_pair_ties_on_tests_but_separates_semantically(self):
        p1, p2 = 0.9147135416666666, 0.7506510416666666
        self.assertEqual(
            terminal_reward("test", solved=False, public_pass_fraction=45 / 46),
            terminal_reward("test", solved=False, public_pass_fraction=45 / 46),
        )
        self.assertGreater(
            terminal_reward("semantic", solved=False, reference_agreement=p1),
            terminal_reward("semantic", solved=False, reference_agreement=p2),
        )

    def test_success_always_dominates_failure(self):
        for arm, metric in (("test", "public_pass_fraction"), ("semantic", "reference_agreement")):
            self.assertGreater(
                terminal_reward(arm, solved=True, **{metric: 0.0}),
                terminal_reward(arm, solved=False, **{metric: 1.0}),
            )

    def test_missing_or_invalid_proxy_fails_closed(self):
        for value in (None, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                terminal_reward("semantic", solved=False, reference_agreement=value)


if __name__ == "__main__":
    unittest.main()
