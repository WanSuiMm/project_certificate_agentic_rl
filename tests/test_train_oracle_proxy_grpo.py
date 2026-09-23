import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from function_swe_executor import CommandExecutor
from train_oracle_proxy_grpo import cache_old_and_reference, rollout, update_policy
from tests.test_function_swe import toy_task


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return torch.tensor([[1, 3]])

    def decode(self, ids, **kwargs):
        return "def double(x):\n    return x * 2\n"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.reference = False

    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, torch.tensor([[2]], device=input_ids.device)), dim=1)

    def forward(self, input_ids, **kwargs):
        scores = torch.zeros_like(self.scores) if self.reference else self.scores
        return SimpleNamespace(logits=scores.view(1, 1, -1).expand(1, input_ids.shape[1], -1))

    @contextmanager
    def disable_adapter(self):
        self.reference = True
        try:
            yield
        finally:
            self.reference = False


class TrainSmokeTests(unittest.TestCase):
    def test_three_step_rollout_and_clipped_update_on_trusted_fixture(self):
        task = toy_task()
        model = FakeModel()
        tokenizer = FakeTokenizer()
        executor = CommandExecutor([sys.executable, str(ROOT / "scripts" / "function_swe_worker.py")])
        config = {"max_edits": 3, "context_tokens": 32,
                  "max_generated_tokens_per_edit": 1, "ppo_epochs": 2,
                  "clip_epsilon": 0.2, "reference_kl_beta": 0.04,
                  "max_grad_norm": 1.0}
        episode = rollout(task, model, tokenizer, executor, device=torch.device("cpu"),
                          config=config, sample=True)
        self.assertEqual(len(episode.actions), 3)
        self.assertTrue(episode.final_score["solved"])
        self.assertEqual(episode.final_score["q"], 1.0)
        self.assertNotIn("probe_cases", tokenizer.last_messages[1]["content"])
        cache_old_and_reference(model, episode.actions, torch.device("cpu"))
        before = model.scores.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        metrics = update_policy(model, optimizer, episode.actions, [1.0] * 3,
                                device=torch.device("cpu"), config=config)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertFalse(torch.equal(before, model.scores.detach()))


if __name__ == "__main__":
    unittest.main()
