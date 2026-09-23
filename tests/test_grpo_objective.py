import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grpo_objective import clipped_grpo_loss, group_advantages


class GrpoObjectiveTests(unittest.TestCase):
    def test_group_advantages_center_and_zero_ties(self):
        a = group_advantages(torch.tensor([0., 0., 0., 0., 0., .5, 0., 1.]), group_size=4)
        self.assertTrue(torch.equal(a[:4], torch.zeros(4)))
        self.assertAlmostEqual(float(a[4:].mean()), 0.0, places=6)
        self.assertGreater(float(a[7]), float(a[5]))

    def test_clipping_blocks_overlarge_positive_ratio(self):
        new = torch.tensor([[0.0]], requires_grad=True)
        old = torch.tensor([[-1.0]])
        ref = new.detach().clone()
        loss, metrics = clipped_grpo_loss(new, old, ref, torch.ones_like(new), torch.ones(1), kl_beta=0)
        self.assertAlmostEqual(float(loss), -1.2, places=6)
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 1.0)
        loss.backward()
        self.assertAlmostEqual(float(new.grad), 0.0, places=6)

    def test_reference_kl_nonnegative_and_masked(self):
        new = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        old = new.detach().clone()
        ref = torch.tensor([[-1.0, -1.0]])
        loss, metrics = clipped_grpo_loss(new, old, ref, torch.tensor([[1., 0.]]), torch.zeros(1))
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["mean_kl"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
