import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import swesmith_agent_worker as worker_module
from swesmith_modal_executor import SWESmithTaskSession


class PersistentWorkerTest(unittest.TestCase):
    def test_one_injection_multiple_public_scores_then_q(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "f.py"
            target.write_text("def f(x):\n    return x\n", encoding="utf-8")
            calls = {"injection": 0, "tests": 0}

            def inject(command, **_kwargs):
                if command[:2] == ["git", "status"]:
                    return SimpleNamespace(returncode=0, stdout=" M f.py\n")
                calls["injection"] += 1
                target.write_text("def f(x):\n    return x - 1\n", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            def public(_selectors):
                calls["tests"] += 1
                return {"valid": True, "exitcode": 1, "passed": 0, "total": 1,
                        "pass_fraction": 0.0}

            task = {"instance_id": "task", "patch": "patch", "FAIL_TO_PASS": ["test_f"],
                    "PASS_TO_PASS": []}
            with (patch.object(worker_module, "ROOT", root),
                  patch.object(worker_module, "patch_location", return_value=("f.py", None)),
                  patch.object(worker_module.subprocess, "run", side_effect=inject),
                  patch.object(worker_module, "tests", side_effect=public),
                  patch.object(worker_module, "run_observer", return_value=["same"] * 256)):
                worker = worker_module.TaskWorker()
                initial = worker.initialize(task, "source_only")
                bank = {"reference": ["same"] * 256, "module": "f",
                        "callable": ["f"], "params": [], "cases": []}
                self.assertEqual(worker.set_q_bank(bank)["status"], "q_bank_ready")
                for value in ("return x + 1", "return x + 2"):
                    result = worker.score({"mode": "score", "instance_id": "task",
                                           "buggy_source_sha256": initial["buggy_source_sha256"],
                                           "source": "def f(x):\n    " + value + "\n",
                                           "include_proxy": False})
                    self.assertEqual(result["status"], "scored")
                    self.assertNotIn("q", result)
                q = worker.score({"mode": "proxy_only", "instance_id": "task",
                                  "buggy_source_sha256": initial["buggy_source_sha256"],
                                  "source": target.read_text(encoding="utf-8")})
                self.assertEqual(q["q"], 1.0)
                self.assertEqual(calls, {"injection": 1, "tests": 2})

    def test_modal_session_sends_json_lines_and_terminates_once(self):
        class FakeStdin:
            def __init__(self):
                self.writes = []

            def write(self, data):
                self.writes.append(json.loads(data.decode()))

            def drain(self):
                pass

        class FakeSandbox:
            def __init__(self):
                self.stdin = FakeStdin()
                self.stdout = iter([
                    json.dumps({"status": "source_only", "instance_id": "task",
                                "buggy_source_sha256": "hash"}) + "\n",
                    json.dumps({"status": "q_bank_ready", "instance_id": "task"}) + "\n",
                    json.dumps({"status": "scored", "instance_id": "task",
                                "public": {}, "p_T": 0.0, "solved": False}) + "\n",
                ])
                self.terminations = 0
                self.detachments = 0

            def terminate(self):
                self.terminations += 1

            def detach(self):
                self.detachments += 1

        sandbox = FakeSandbox()
        executor = SimpleNamespace(
            modal=SimpleNamespace(Sandbox=SimpleNamespace(create=lambda *_a, **_kw: sandbox)),
            app=object(), _image=lambda _name: object(),
        )
        task = {"instance_id": "task", "image_name": "image"}
        with SWESmithTaskSession(executor, task) as session:
            session.initialize(mode="source_only")
            session.set_q_bank({"reference": [0] * 256})
            result = session.call(mode="score", source="program", include_proxy=False)
            self.assertEqual(result["status"], "scored")
        self.assertEqual([row["mode"] for row in sandbox.stdin.writes],
                         ["source_only", "set_q_bank", "score"])
        self.assertEqual(sandbox.terminations, 1)
        self.assertEqual(sandbox.detachments, 1)


if __name__ == "__main__":
    unittest.main()
