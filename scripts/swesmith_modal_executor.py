"""Official SWE-smith Modal execution, including one reusable sandbox per task."""

from __future__ import annotations

import json
from pathlib import Path


class ScorerError(RuntimeError):
    pass


class WorkerEndedError(ScorerError):
    """The persistent sandbox stopped before returning one observation."""


class SWESmithTaskSession:
    """One JSON-lines worker and checkout for all observations of one task."""

    def __init__(self, executor, task: dict) -> None:
        self.executor = executor
        self.task = task
        self.sandbox = None
        self.output = None
        self.buggy_source_sha256 = None
        self.worker_ended = False

    def __enter__(self):
        self.sandbox = self.executor.modal.Sandbox.create(
            "python", "-u", "/opt/swesmith_agent_worker.py", "--serve",
            app=self.executor.app, image=self.executor._image(self.task["image_name"]),
            timeout=7200, cpu=1, memory=2048, block_network=True,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.output = iter(self.sandbox.stdout)
        return self

    def __exit__(self, *_exc) -> None:
        if self.sandbox is not None:
            sandbox, self.sandbox = self.sandbox, None
            try:
                if not self.worker_ended:
                    sandbox.terminate()
            finally:
                sandbox.detach()

    def _request(self, request: dict) -> dict:
        if self.sandbox is None:
            raise ScorerError("task session is closed")
        self.sandbox.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
        self.sandbox.stdin.drain()
        try:
            result = json.loads(next(self.output))
        except (StopIteration, json.JSONDecodeError) as exc:
            self.worker_ended = True
            raise WorkerEndedError("task worker ended or returned non-JSON") from exc
        if result.get("status") in {"worker_error", "invalid_test_observation"}:
            raise ScorerError(f"invalid scorer result: {result}")
        if result.get("instance_id") != self.task["instance_id"]:
            raise ScorerError("task worker returned another task")
        return result

    def initialize(self, *, mode: str = "init") -> dict:
        if self.buggy_source_sha256 is not None or mode not in {"init", "source_only"}:
            raise ScorerError("task session already initialized or bad init mode")
        result = self._request({"mode": mode, "task": self.task})
        self.buggy_source_sha256 = result["buggy_source_sha256"]
        return result

    def set_q_bank(self, bank: dict) -> None:
        if self.buggy_source_sha256 is None:
            raise ScorerError("task session not initialized")
        result = self._request({"mode": "set_q_bank", "q_bank": bank})
        if result.get("status") != "q_bank_ready":
            raise ScorerError("worker did not accept q bank")

    def call(self, *, mode: str, source: str, q_bank: dict | None = None,
             include_proxy: bool = False) -> dict:
        if self.buggy_source_sha256 is None or mode not in {"score", "proxy_only"}:
            raise ScorerError("task session not initialized or bad scoring mode")
        return self._request({
            "mode": mode, "instance_id": self.task["instance_id"],
            "source": source, "buggy_source_sha256": self.buggy_source_sha256,
            "q_bank": q_bank, "include_proxy": include_proxy,
        })


class SWESmithModalExecutor:
    def __init__(self) -> None:
        import modal

        self.modal = modal
        self.app = modal.App.lookup("certificate-swesmith-agent-rl-v01", create_if_missing=True)
        self.scripts = Path(__file__).resolve().parent
        self.images = {}

    def _image(self, name: str):
        if name not in self.images:
            image = self.modal.Image.from_registry(name)
            for filename in ("swesmith_agent_worker.py", "swesmith_q_qualifier_worker.py"):
                image = image.add_local_file(str(self.scripts / filename), remote_path=f"/opt/{filename}")
            self.images[name] = image
        return self.images[name]

    def task_session(self, task: dict) -> SWESmithTaskSession:
        return SWESmithTaskSession(self, task)

    def call(self, task: dict, *, mode: str, source: str | None = None,
             buggy_source_sha256: str | None = None, q_bank: dict | None = None,
             include_proxy: bool = False) -> dict:
        request = {"mode": mode, "task": task}
        if mode in {"score", "proxy_only"}:
            request.update({"source": source, "buggy_source_sha256": buggy_source_sha256,
                            "q_bank": q_bank, "include_proxy": include_proxy})
        sandbox = self.modal.Sandbox.create(
            "python", "/opt/swesmith_agent_worker.py", app=self.app,
            image=self._image(task["image_name"]), timeout=600, cpu=1,
            memory=2048, block_network=True,
        )
        try:
            sandbox.stdin.write(json.dumps(request, ensure_ascii=False).encode())
            sandbox.stdin.write_eof()
            sandbox.stdin.drain()
            sandbox.wait()
            stdout = sandbox.stdout.read()
            stderr = sandbox.stderr.read()
            if sandbox.returncode:
                raise ScorerError(f"worker exited {sandbox.returncode}: {stderr[-500:]}")
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise ScorerError(f"worker returned non-JSON: {stdout[-500:]}") from exc
            if result.get("status") in {"worker_error", "invalid_test_observation"}:
                raise ScorerError(f"invalid scorer result: {result}")
            return result
        finally:
            sandbox.terminate()
            sandbox.detach()
