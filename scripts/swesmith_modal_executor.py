"""Fresh official SWE-smith image for each agent initialization or score."""

from __future__ import annotations

import json
from pathlib import Path


class ScorerError(RuntimeError):
    pass


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
