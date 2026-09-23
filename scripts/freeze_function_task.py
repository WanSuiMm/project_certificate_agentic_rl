#!/usr/bin/env python3
"""Qualify and freeze one pure-function task INSIDE an isolated executor.

Input spec contains sources, official public/hidden cases with expected
observations, and 256 probe inputs. Only probe expectations come from the clean
reference. The worker must not be run on a machine containing secrets.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from function_swe import TaskError, canonical_json, load_task, observe, score_source, sha256_text


def _load_function(source: str, target: str) -> Any:
    namespace: dict[str, Any] = {"__name__": "__function_swe_reference__"}
    exec(compile(source, "<task-source>", "exec"), namespace)
    function = namespace.get(target)
    if not callable(function):
        raise TaskError("target is not callable")
    return function


def freeze_spec(spec: dict[str, Any], *, expected_probes: int = 256) -> dict[str, Any]:
    required = ("task_id", "issue", "target_function", "buggy_source", "reference_source",
                "public_cases", "hidden_cases", "probe_inputs", "provenance")
    if any(key not in spec for key in required):
        raise TaskError("incomplete task spec")
    if len(spec["probe_inputs"]) != expected_probes:
        raise TaskError("probe input count mismatch")
    for key in ("public_cases", "hidden_cases"):
        if not spec[key] or any("expect" not in case for case in spec[key]):
            raise TaskError(f"{key} must contain official expected observations")
    input_keys = [canonical_json({"args": c["args"], "kwargs": c["kwargs"]})
                  for key in ("public_cases", "hidden_cases", "probe_inputs") for c in spec[key]]
    if len(set(input_keys)) != len(input_keys):
        raise TaskError("public, hidden, and probe inputs must be disjoint")
    target = spec["target_function"]
    for key in ("buggy_source", "reference_source"):
        functions = [n.name for n in ast.parse(spec[key]).body if isinstance(n, ast.FunctionDef)]
        if functions.count(target) != 1:
            raise TaskError(f"{key} must contain one top-level target function")
    reference = _load_function(spec["reference_source"], target)
    for key in ("public_cases", "hidden_cases"):
        for case in spec[key]:
            if observe(reference, case) != case["expect"]:
                raise TaskError(f"clean reference fails a supplied {key} expectation")
    probes = []
    for case in spec["probe_inputs"]:
        first = observe(reference, case)
        if observe(reference, case) != first:
            raise TaskError("reference probe is nondeterministic")
        probes.append({"args": case["args"], "kwargs": case["kwargs"], "expect": first})
    task = {
        "task_id": spec["task_id"], "issue": spec["issue"],
        "target_function": target, "buggy_source": spec["buggy_source"],
        "buggy_sha256": sha256_text(spec["buggy_source"]),
        "reference_sha256": sha256_text(spec["reference_source"]),
        "public_cases": spec["public_cases"], "hidden_cases": spec["hidden_cases"],
        "probe_cases": probes, "provenance": spec["provenance"],
    }
    initial = score_source(task, task["buggy_source"], terminal=True)
    if initial["q"] >= 1.0 or initial["solved"]:
        raise TaskError("initial program has no reference headroom or is already solved")
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    task = freeze_spec(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    load_task(args.output)
    print(f"frozen {task['task_id']} at {args.output}")


if __name__ == "__main__":
    main()
