import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from function_swe import score_source
from function_swe_executor import ExecutorError
from modal_function_swe_executor import ModalSandboxExecutor
from smoke_modal_function_swe import run_smoke, toy_task


class FakeInput:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data):
        self.data += data

    def write_eof(self):
        self.closed = True

    def drain(self):
        pass


class FakeImage:
    @classmethod
    def debian_slim(cls):
        return cls()

    def add_local_file(self, local, *, remote_path):
        self.files = getattr(self, "files", []) + [(local, remote_path)]
        return self


class FakeSandbox:
    calls = []
    corrupt = False

    @classmethod
    def create(cls, *command, **options):
        sandbox = cls()
        sandbox.command = command
        sandbox.options = options
        sandbox.stdin = FakeInput()
        sandbox.stdout = io.BytesIO()
        sandbox.stderr = io.BytesIO()
        sandbox.returncode = None
        sandbox.terminated = sandbox.detached = False
        cls.calls.append(sandbox)
        return sandbox

    def wait(self):
        request = json.loads(self.stdin.data)
        if self.corrupt:
            self.stdout = io.BytesIO(b"not JSON")
        else:
            result = score_source(request["task"], request["source"],
                                  terminal=request["terminal"],
                                  include_proxy=request["include_proxy"])
            self.stdout = io.BytesIO(json.dumps(result).encode())
        self.returncode = 0

    def terminate(self):
        self.terminated = True

    def detach(self):
        self.detached = True


class ModalExecutorTests(unittest.TestCase):
    def setUp(self):
        FakeSandbox.calls = []
        FakeSandbox.corrupt = False
        self.modal = SimpleNamespace(
            App=SimpleNamespace(lookup=lambda *args, **kwargs: object()),
            Image=FakeImage, Sandbox=FakeSandbox,
        )

    def test_trusted_smoke_uses_fresh_network_blocked_sandboxes(self):
        executor = ModalSandboxExecutor(modal_module=self.modal)
        run_smoke(executor)
        self.assertEqual(len(FakeSandbox.calls), 2)
        self.assertIsNot(FakeSandbox.calls[0], FakeSandbox.calls[1])
        for sandbox in FakeSandbox.calls:
            self.assertEqual(sandbox.command, ("python", "/opt/function_swe/function_swe_worker.py"))
            self.assertTrue(sandbox.options["block_network"])
            self.assertNotIn("secrets", sandbox.options)
            self.assertNotIn("volumes", sandbox.options)
            self.assertEqual(sandbox.options["memory"], (256, 1024))
            self.assertTrue(sandbox.stdin.closed)
            self.assertTrue(sandbox.terminated and sandbox.detached)

    def test_invalid_worker_output_fails_closed_and_cleans_up(self):
        executor = ModalSandboxExecutor(modal_module=self.modal)
        FakeSandbox.corrupt = True
        with self.assertRaises(ExecutorError):
            executor.score(toy_task(), "def double(x): return x * 2", terminal=True)
        self.assertTrue(FakeSandbox.calls[0].terminated)
        self.assertTrue(FakeSandbox.calls[0].detached)


if __name__ == "__main__":
    unittest.main()
