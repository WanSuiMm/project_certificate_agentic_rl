import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_swesmith_oracle_credit as runner


class OracleRunnerTest(unittest.TestCase):
    def test_ready_trajectories_advance_before_one_slow_public_test(self):
        source = "def f(x):\n    return 1\n"
        task_id = "task"
        item = {"task": {"instance_id": task_id},
                "q_bank": {"callable": ["f"], "module": "f", "reference": "not_actor_visible"}}
        public = {"status": "scored", "p_T": 0.0, "solved": False,
                  "public": {"valid": True, "passed": 0, "total": 1, "exitcode": 1}}
        frozen = [{"instance_id": task_id, "sample": i, "completion": "return 1",
                   "status": "scored", "final_source_sha256": runner.sha256(source),
                   "score": {**public, "q": i / 16}} for i in range(16)]
        candidates = {(task_id, i): {"instance_id": task_id, "sample": i,
                                     "source": source, "source_sha256": runner.sha256(source),
                                     "invalid_body": False, "p": 0.0, "q": i / 16,
                                     **runner.public_fields(public)} for i in range(16)}

        class Session:
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def initialize(self, *, mode):
                assert mode == "source_only"
                return {"buggy_source": source, "buggy_source_sha256": runner.sha256(source)}

        class Executor:
            def task_session(self, task): return Session()

        slow_done, early_generation = threading.Event(), threading.Event()

        def score(pending, _sessions, _executor, _task):
            if pending["sample"] == 0 and pending["replicate"] == 0 and pending["step"] == 2:
                time.sleep(0.4)
                slow_done.set()
            else:
                time.sleep(0.002)
            return {**pending, **runner.public_fields(public), "public_seconds": 0.0}

        def prompt(safe_item, current, _feedback):
            assert "reference" not in safe_item["q_bank"]
            return current

        def generate(_model, _tokenizer, prompts, *, config, seed):
            if any("return 2" in prompt for prompt in prompts) and not slow_done.is_set():
                early_generation.set()
            return [{"completion": "return 2", "generated_tokens": 2, "hit_token_cap": False}
                    for _ in prompts]

        config = {"continuations_per_candidate": 4, "generation_batch_size": 8,
                  "sandbox_workers": 4, "seed": 0}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "action_batches").mkdir()
            states, actions, outcomes = {}, {}, {}
            with patch.object(runner, "score_pending", score), patch.object(runner, "agent_prompt", prompt), \
                    patch.object(runner, "generate_batch", generate):
                runner.run_task(item, frozen, config=config, output=output, executor=Executor(),
                                tokenizer=None, model=None, candidates=candidates,
                                actions=actions, states=states, outcomes=outcomes,
                                progress=lambda *_args: None)
            self.assertTrue(early_generation.is_set(), "one slow P2 must not block other P3 actions")
            self.assertEqual((len(states), len(outcomes)), (448, 64))
            runner.validate_state_chain(candidates, actions, states, outcomes)

    def test_dead_worker_falls_back_to_one_shot_without_reusing_session(self):
        class DeadSession:
            def call(self, **_kwargs):
                raise runner.WorkerEndedError("worker ended")

        class Executor:
            calls = 0

            def call(self, task, *, mode, source, include_proxy):
                self.calls += 1
                self.assertion = (task["instance_id"], mode, source, include_proxy)
                return {"status": "scored", "p_T": 0.5, "solved": False,
                        "public": {"valid": True, "passed": 1, "total": 2, "exitcode": 1}}

        sessions = Queue()
        sessions.put(DeadSession())
        executor = Executor()
        pending = {"source": "def f():\n    return 1\n", "instance_id": "task"}
        for _ in range(2):
            result = runner.score_pending(pending, sessions, executor, {"instance_id": "task"})
            self.assertEqual(result["p_T"], 0.5)
            self.assertNotIn("q", result)
        self.assertEqual(executor.calls, 2)
        self.assertEqual(executor.assertion, ("task", "score", pending["source"], False))
        self.assertIsNone(sessions.get_nowait())

    def test_failed_checkpoint_import_preserves_old_run_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / "old", root / "new"
            (old / "action_batches").mkdir(parents=True)
            (new / "action_batches").mkdir(parents=True)
            receipt = {"config": {"protocol": "test"}, "task_ids": ["t"],
                       "selection_sha256": "a", "census_sha256": "b",
                       "tasks_sha256": "c", "q_results_sha256": "d"}
            old_run = {**receipt, "status": "failed", "code_sha256": {"runner.py": "old"},
                       "completed_edit_steps": 1, "completed_endpoints": 0}
            (old / "run.json").write_text(json.dumps(old_run))
            for name in ("frozen_p1_inputs.jsonl", "candidates.jsonl", "states.jsonl", "outcomes.jsonl"):
                (old / name).write_text('{"sample": 0}\n')
            (old / "action_batches" / "one.json").write_text('{"records": []}\n')
            summary = runner.import_checkpoint(old, new, receipt)
            self.assertEqual(summary["completed_edit_steps"], 1)
            self.assertEqual((new / "states.jsonl").read_bytes(), (old / "states.jsonl").read_bytes())
            self.assertEqual((new / "action_batches" / "one.json").read_bytes(),
                             (old / "action_batches" / "one.json").read_bytes())
            manifest = json.loads((new / "import_manifest.json").read_text())
            self.assertEqual(manifest["source_run_sha256"], runner.digest(old / "run.json"))
            self.assertEqual(manifest["evidence_sha256"]["states.jsonl"], runner.digest(old / "states.jsonl"))
            with self.assertRaisesRegex(ValueError, "not empty"):
                runner.import_checkpoint(old, new, receipt)

    def test_frozen_invalid_action_keeps_source_and_gets_actual_proxy(self):
        source = "def f(x):\n    return x\n"
        frozen = {"instance_id": "task", "sample": 0, "completion": "if ???",
                  "status": "invalid_body", "final_source_sha256": runner.sha256(source),
                  "score": {"p_T": 0, "q": 0, "solved": False}}

        class Session:
            def set_q_bank(self, bank):
                pass

            def call(self, *, mode, source, include_proxy=False):
                if mode == "proxy_only":
                    return {"status": "q_scored", "q": 0.75}
                return {"status": "scored", "p_T": 0.5, "solved": False,
                        "public": {"valid": True, "passed": 1, "total": 2, "exitcode": 1}}

        item = {"task": {"instance_id": "task"}, "q_bank": {"callable": ["f"]}}
        row = runner.prepare_candidates(item, [frozen], Session(), {"buggy_source": source})[0]
        self.assertEqual((row["p"], row["q"]), (0.5, 0.75))
        self.assertTrue(row["invalid_body"])
        self.assertEqual(row["source"], source)
        corrupt = {**frozen, "final_source_sha256": "bad"}
        with self.assertRaises(ValueError):
            runner.reconstruct_p1(source, ["f"], corrupt)

    def test_closed_loop_four_forks_durable_actions_and_resume(self):
        source = "def f(x):\n    return x\n"
        events = []
        item = {"task": {"instance_id": "task"}, "q_bank": {"callable": ["f"], "module": "f", "reference": "secret"}}
        public = {"status": "scored", "p_T": 0.0, "solved": False,
                  "public": {"valid": True, "passed": 0, "total": 1, "exitcode": 1}}

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def initialize(self, *, mode):
                return {"gold_self_consistent": True, "buggy_source": source,
                        "buggy_source_sha256": runner.sha256(source)}

            def call(self, *, mode, source, include_proxy=False):
                self.assert_mode = mode
                assert mode == "score" and not include_proxy
                events.append("public")
                return public

        class Executor:
            def task_session(self, task):
                return Session()

        def prompt(safe_item, current, feedback):
            assert "reference" not in safe_item["q_bank"]
            assert feedback == "Public tests: 0/1 passed; exitcode=1."
            return current + feedback

        def generate(_model, _tokenizer, prompts, *, config, seed):
            events.append("generate")
            return [{"completion": "return x", "generated_tokens": 2, "hit_token_cap": False} for _ in prompts]

        frozen = [{"instance_id": "task", "sample": i, "completion": "return x", "status": "scored",
                   "final_source_sha256": runner.sha256(source), "score": {**public, "q": i / 16}} for i in range(16)]
        config = {"continuations_per_candidate": 4, "generation_batch_size": 8, "sandbox_workers": 2, "seed": 0}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "action_batches").mkdir()
            candidates, actions, states, outcomes = {}, {}, {}, {}

            def interrupt_progress(_task, _step):
                if len(states) == 10:
                    raise RuntimeError("simulated interruption")

            with patch.object(runner, "generate_batch", generate), patch.object(runner, "agent_prompt", prompt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.run_task(item, frozen, config=config, output=output, executor=Executor(),
                                    tokenizer=None, model=None, candidates=candidates, actions=actions,
                                    states=states, outcomes=outcomes, progress=interrupt_progress)
                # Reload the same files a restarted process uses.
                candidates = runner.unique_rows(runner.read_rows(output / "candidates.jsonl"), runner.candidate_key)
                states = runner.unique_rows(runner.read_rows(output / "states.jsonl"), runner.state_key)
                actions = runner.unique_rows([row for p in (output / "action_batches").glob("*.json")
                                               for row in json.loads(p.read_text())["records"]], runner.state_key)
                runner.validate_state_chain(candidates, actions, states, outcomes)
                runner.run_task(item, frozen, config=config, output=output, executor=Executor(),
                                tokenizer=None, model=None, candidates=candidates, actions=actions,
                                states=states, outcomes=outcomes, progress=lambda *_a: None)
                runner.validate_state_chain(candidates, actions, states, outcomes)
            self.assertEqual((len(candidates), len(actions), len(states), len(outcomes)), (16, 448, 448, 64))
            self.assertEqual(events.count("generate"), 56)
            self.assertTrue(all("q" not in row for row in states.values()))
            self.assertEqual({key[2] for key in outcomes}, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
