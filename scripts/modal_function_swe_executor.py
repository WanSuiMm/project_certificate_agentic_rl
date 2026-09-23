"""Fresh Modal Sandbox for each Function-SWE scoring request.

The Modal client stays with the trainer; candidate source only enters a new,
network-blocked Sandbox with no mounted secrets or persistent volumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from function_swe_executor import ExecutorError, validate_score


class ModalSandboxExecutor:
    def __init__(self, *, app_name: str = "certificate-function-swe-v01",
                 timeout_seconds: int = 45, modal_module: Any = None) -> None:
        if timeout_seconds < 1:
            raise ValueError("timeout must be positive")
        if modal_module is None:
            try:
                import modal as modal_module
            except ImportError as exc:
                raise RuntimeError("install the Modal Python SDK on the training host") from exc
        self.modal = modal_module
        self.timeout_seconds = timeout_seconds
        self.app = self.modal.App.lookup(app_name, create_if_missing=True)
        scripts = Path(__file__).resolve().parent
        self.image = (
            self.modal.Image.debian_slim()
            .add_local_file(str(scripts / "function_swe.py"), remote_path="/opt/function_swe/function_swe.py")
            .add_local_file(str(scripts / "function_swe_worker.py"), remote_path="/opt/function_swe/function_swe_worker.py")
        )

    def score(self, task: dict[str, Any], source: str, *, terminal: bool,
              include_proxy: bool = True) -> dict[str, Any]:
        request = {"task": task, "source": source, "terminal": terminal,
                   "include_proxy": include_proxy}
        sandbox = None
        try:
            sandbox = self.modal.Sandbox.create(
                "python", "/opt/function_swe/function_swe_worker.py",
                app=self.app, image=self.image, timeout=self.timeout_seconds,
                cpu=(0.25, 1.0), memory=(256, 1024), block_network=True,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            sandbox.stdin.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))
            sandbox.stdin.write_eof()
            sandbox.stdin.drain()
            sandbox.wait()
            stdout = _read_bounded(sandbox.stdout, 64 * 1024)
            stderr = _read_bounded(sandbox.stderr, 4 * 1024)
            if sandbox.returncode != 0:
                raise ExecutorError(
                    f"Modal scorer failed (exit {sandbox.returncode}): "
                    f"{_display(stderr)[-300:]}"
                )
            try:
                result = json.loads(_display(stdout))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ExecutorError("Modal scorer returned non-JSON output") from exc
            return validate_score(result, terminal=terminal, include_proxy=include_proxy)
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"Modal scoring transport failed: {type(exc).__name__}") from exc
        finally:
            if sandbox is not None:
                try:
                    sandbox.terminate()
                finally:
                    sandbox.detach()


def _display(data: bytes | str) -> str:
    return data.decode("utf-8") if isinstance(data, bytes) else data


def _read_bounded(stream: Any, limit: int) -> str:
    chunks = []
    size = 0
    for chunk in stream:
        value = _display(chunk)
        size += len(value.encode("utf-8"))
        if size > limit:
            raise ExecutorError("Modal scorer output exceeded its size limit")
        chunks.append(value)
    return "".join(chunks)
