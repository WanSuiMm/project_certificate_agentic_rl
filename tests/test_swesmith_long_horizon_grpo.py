import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_swesmith_trajectory_grpo_signal import summarize
from train_swesmith_long_horizon_grpo import (
    Trajectory, action_advantages, stepwise_group_advantages,
    stepwise_returns, trajectory_reward, rollout_group,
)


class LongHorizonGRPOTest(unittest.TestCase):
    def test_rollout_samples_all_actions_before_hidden_q(self):
        source = "def f():\n    return 0\n"
        events = []

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def initialize(self, *, mode):
                events.append(mode)
                return {"gold_self_consistent": True, "buggy_source": source}

            def call(self, *, mode, source, include_proxy=False):
                events.append(mode)
                if mode == "proxy_only":
                    return {"status": "q_scored", "q": 0.25}
                assert not include_proxy
                return {"public": {"valid": True, "passed": 0, "total": 1,
                                   "exitcode": 1}, "p_T": 0.0,
                        "solved": False, "status": "scored"}

            def set_q_bank(self, bank):
                events.append("set_q_bank")

        class Executor:
            def task_session(self, task):
                return Session()

        def fake_sample_step(**kwargs):
            events.append("sample")
            prior = kwargs["prior"]
            state = {**prior, "feedback_given": "Public tests: 0/1 passed; exitcode=1."}
            return state, object()

        item = {"task": {"instance_id": "task"},
                "q_bank": {"callable": ["f"], "module": "module"}}
        with patch("train_swesmith_long_horizon_grpo.sample_step", fake_sample_step):
            episodes = rollout_group(item, None, None, Executor(), config={"seed": 0},
                                     device=torch.device("cpu"), update=1,
                                     group_size=2, sample=True, q_scope="all")
        self.assertEqual(len(episodes), 2)
        self.assertEqual(events.count("sample"), 16)
        self.assertLess(max(i for i, event in enumerate(events) if event == "sample"),
                        events.index("set_q_bank"))
        self.assertEqual(events.count("proxy_only"), 1)
        self.assertTrue(all(state["q"] == 0.25 for episode in episodes
                            for state in episode.states))

    def test_whole_trajectory_reward_repeats_one_advantage_eight_times(self):
        episodes = []
        for sample in range(16):
            states = [{"p_T": 0.0, "q": sample / 16, "solved": False}
                      for _ in range(9)]
            episodes.append(Trajectory(states, [object()] * 8))
        rewards, advantages = action_advantages(episodes, "semantic", "trajectory", 0.5)
        self.assertEqual(len(rewards), 16)
        self.assertEqual(len(advantages), 128)
        self.assertEqual(len(set(rewards)), 16)
        self.assertTrue(all(len(set(advantages[i * 8:(i + 1) * 8])) == 1
                            for i in range(16)))
        test_rewards, test_advantages = action_advantages(episodes, "test", "trajectory", 0.5)
        self.assertEqual(set(test_rewards), {0.0})
        self.assertEqual(set(test_advantages), {0.0})

    def test_stepwise_credit_changes_with_previous_potential(self):
        potentials = [0.0, 0.2, 0.1, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8]
        returns = stepwise_returns(False, potentials)
        self.assertEqual(len(returns), 8)
        self.assertAlmostEqual(returns[0], 0.4)
        self.assertAlmostEqual(returns[1], 0.3)
        self.assertAlmostEqual(trajectory_reward(True, 0.8), 1.4)
        advantages = stepwise_group_advantages([returns, [0.0] * 8])
        self.assertEqual(len(advantages), 2)
        self.assertNotEqual(advantages[0][0], advantages[1][0])

    def test_census_counts_semantic_rescue_of_tied_test_group(self):
        states, observations = [], []
        for trajectory in [None, *range(16)]:
            steps = [0] if trajectory is None else range(1, 9)
            for step in steps:
                key = f"state-{trajectory}-{step}"
                q = 0.0 if trajectory is None else trajectory / 16
                state = {"instance_id": "task", "trajectory_id": trajectory,
                         "step": step, "source_sha256": key,
                         "p_T": 0.0, "solved": False}
                states.append(state)
                observations.append({**state, "status": "q_scored", "q": q})
        result = summarize(states, observations, ["task"])
        self.assertEqual(result["semantic_rescue_tasks"], 1)
        self.assertEqual(result["test_informative_tasks"], 0)
        self.assertEqual(result["semantic_informative_tasks"], 1)
        self.assertEqual(result["q_changes_before_public_trajectories"], 15)


if __name__ == "__main__":
    unittest.main()
