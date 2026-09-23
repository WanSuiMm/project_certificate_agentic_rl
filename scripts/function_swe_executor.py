"""JSON-over-stdin bridge to an isolated function scorer."""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any, Sequence


class ExecutorError(RuntimeError):
    """Worker/transport fault: abort the run, do not turn into reward zero."""


def validate_score(result: Any, *, terminal: bool, include_proxy: bool) -> dict[str, Any]:
    required = {"public_passed", "public_total", "p_T"}
    if terminal:
        required |= {"hidden_passed", "hidden_total", "solved"}
        if include_proxy:
            required.add("q")
    if not isinstance(result, dict) or not required <= result.keys():
        raise ExecutorError("isolated worker returned incomplete score")
    if (not isinstance(result["public_passed"], int)
            or not isinstance(result["public_total"], int)
            or not 0 <= result["public_passed"] <= result["public_total"]
            or result["public_total"] < 1):
        raise ExecutorError("isolated worker returned invalid public counts")
    for key in ("p_T", "q"):
        if key in result and (not isinstance(result[key], (int, float))
                              or not math.isfinite(result[key]) or not 0 <= result[key] <= 1):
            raise ExecutorError(f"isolated worker returned invalid {key}")
    if abs(result["p_T"] - result["public_passed"] / result["public_total"]) > 1e-9:
        raise ExecutorError("isolated worker returned inconsistent public score")
    if terminal and (not isinstance(result["solved"], bool)
                     or not isinstance(result["hidden_passed"], int)
                     or not isinstance(result["hidden_total"], int)
                     or not 0 <= result["hidden_passed"] <= result["hidden_total"]
                     or result["hidden_total"] < 1):
        raise ExecutorError("isolated worker returned invalid terminal score")
    if terminal and result["solved"] != (
        result["public_passed"] == result["public_total"]
        and result["hidden_passed"] == result["hidden_total"]
    ):
        raise ExecutorError("isolated worker returned inconsistent solved flag")
    return result


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
        return validate_score(result, terminal=terminal, include_proxy=include_proxy)
