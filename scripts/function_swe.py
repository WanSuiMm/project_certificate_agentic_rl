"""Pure-function task contract and edit operations for the RL survival run.

Candidate code is deliberately *not* executed by the agent/controller here.
``score_source`` belongs only in an isolated worker process/container.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class TaskError(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_task(path: Path, *, expected_probes: int = 256) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "task_id", "issue", "target_function", "buggy_source", "buggy_sha256",
        "reference_sha256", "public_cases", "hidden_cases", "probe_cases",
    )
    missing = [key for key in required if key not in task]
    if missing:
        raise TaskError(f"missing task fields: {missing}")
    if not isinstance(task["task_id"], str) or not task["task_id"]:
        raise TaskError("task_id must be nonempty")
    if not isinstance(task["issue"], str) or not task["issue"]:
        raise TaskError("issue must be nonempty")
    if sha256_text(task["buggy_source"]) != task["buggy_sha256"]:
        raise TaskError("buggy source hash mismatch")
    if not isinstance(task["reference_sha256"], str) or len(task["reference_sha256"]) != 64:
        raise TaskError("reference_sha256 must be a SHA-256 digest")
    functions = [node.name for node in ast.parse(task["buggy_source"]).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions.count(task["target_function"]) != 1:
        raise TaskError("target must be one top-level function")
    for key, expected_count in (("public_cases", 1), ("hidden_cases", 1), ("probe_cases", expected_probes)):
        cases = task[key]
        if not isinstance(cases, list) or len(cases) < expected_count or (key == "probe_cases" and len(cases) != expected_count):
            raise TaskError(f"{key} has wrong count")
        for case in cases:
            if not isinstance(case.get("args"), list) or not isinstance(case.get("kwargs"), dict):
                raise TaskError(f"{key} requires args and kwargs")
            if "expect" not in case:
                raise TaskError(f"{key} requires frozen expected observation")
    return task


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def replace_top_level_function(source: str, target: str, completion: str) -> str:
    """Apply exactly one function-definition edit; reject free-form shell/code."""
    replacement = strip_code_fence(completion)
    try:
        new_tree = ast.parse(replacement)
        old_tree = ast.parse(source)
    except SyntaxError as exc:
        raise TaskError(f"unparseable edit: {exc.msg}") from exc
    if len(new_tree.body) != 1 or not isinstance(new_tree.body[0], ast.FunctionDef):
        raise TaskError("edit must contain exactly one synchronous function definition")
    new_node = new_tree.body[0]
    if new_node.name != target:
        raise TaskError("edit changes the wrong function")
    matches = [node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == target]
    if len(matches) != 1:
        raise TaskError("source has no unique target function")
    old_node = matches[0]
    first_line = min([old_node.lineno, *(dec.lineno for dec in old_node.decorator_list)])
    lines = source.splitlines(keepends=True)
    before = "".join(lines[: first_line - 1])
    after = "".join(lines[old_node.end_lineno :])
    edited = before + replacement.rstrip() + "\n" + after
    ast.parse(edited)
    return edited


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_value(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _encode_value(item) for key, item in value.items()}
    raise TaskError(f"unsupported observation value: {type(value).__name__}")


def observe(function: Any, case: dict[str, Any]) -> dict[str, Any]:
    try:
        value = function(*_decode_value(case["args"]), **_decode_value(case["kwargs"]))
        return {"kind": "return", "value": _encode_value(value)}
    except Exception as exc:
        return {"kind": "exception", "type": type(exc).__name__}


def score_source(task: dict[str, Any], source: str, *, terminal: bool,
                 include_proxy: bool = True) -> dict[str, Any]:
    """Execute inside the sandbox ONLY. Infrastructure failure is not a reward."""
    try:
        namespace: dict[str, Any] = {"__name__": "__function_swe_candidate__"}
        exec(compile(source, "<candidate>", "exec"), namespace)
        function = namespace[task["target_function"]]
    except (SyntaxError, NameError, KeyError, ImportError) as exc:
        result = {"candidate_invalid": type(exc).__name__, "public_passed": 0,
                  "public_total": len(task["public_cases"]), "p_T": 0.0}
        if terminal:
            result.update({"hidden_passed": 0, "hidden_total": len(task["hidden_cases"]),
                           "solved": False})
            if include_proxy:
                result["q"] = 0.0
        return result
    public_passed = sum(observe(function, case) == case["expect"] for case in task["public_cases"])
    result: dict[str, Any] = {
        "candidate_invalid": None,
        "public_passed": public_passed,
        "public_total": len(task["public_cases"]),
        "p_T": public_passed / len(task["public_cases"]),
    }
    if terminal:
        hidden_passed = sum(observe(function, case) == case["expect"] for case in task["hidden_cases"])
        result.update({
            "hidden_passed": hidden_passed,
            "hidden_total": len(task["hidden_cases"]),
            "solved": public_passed == len(task["public_cases"]) and hidden_passed == len(task["hidden_cases"]),
        })
        if include_proxy:
            probe_agree = sum(observe(function, case) == case["expect"] for case in task["probe_cases"])
            result["q"] = probe_agree / len(task["probe_cases"])
    return result


def public_prompt(task: dict[str, Any], source: str, feedback: str) -> str:
    public = [{"args": c["args"], "kwargs": c["kwargs"], "expect": c["expect"]} for c in task["public_cases"]]
    return (
        "Repair one pure Python function. Output only its full `def` block, no explanation.\n"
        f"Issue: {task['issue']}\nTarget: {task['target_function']}\n"
        f"Current module:\n```python\n{source}\n```\n"
        f"Visible tests: {canonical_json(public)}\nPrevious public feedback: {feedback}"
    )
