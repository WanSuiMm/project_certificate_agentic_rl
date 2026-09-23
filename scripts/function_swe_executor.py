"""JSON-over-stdin bridge to an isolated function scorer."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Sequence


class ExecutorError(RuntimeError):
    """Worker/transport fault: abort the run, do not turn into reward zero."""


class CommandExecutor:
    def __init__(self, command: Sequence[str], *, timeout_seconds: int = 30) -> None:
        if not command:
            raise ValueError("sandbox command cannot be empty")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds

    def score(self, task: dict[str, Any], source: str, *, terminal: bool,
              include_proxy: bool = True) -> dict[str, Any]:
        request = {"task": task, "source": source, "terminal": terminal,
                   "include_proxy": include_proxy}
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorError(f"isolated worker unavailable: {type(exc).__name__}") from exc
        if completed.returncode != 0:
            raise ExecutorError(
                f"isolated worker failed (exit {completed.returncode}): "
                f"{completed.stderr[-300:]}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorError("isolated worker returned non-JSON output") from exc
        required = {"public_passed", "public_total", "p_T"}
        if terminal:
            required |= {"hidden_passed", "hidden_total", "solved"}
            if include_proxy:
                required.add("q")
        if not isinstance(result, dict) or not required <= result.keys():
            raise ExecutorError("isolated worker returned incomplete score")
        return result
