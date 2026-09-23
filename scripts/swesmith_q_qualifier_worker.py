"""Qualify a SWE-smith task for automatic reference-behavior q inside its image.

No official test input is read. The input bank derives only from the patched
callable's signature, defaults, parameter names, and module string constants.
Run this file only inside an isolated sandbox with no secrets or network.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys


ROOT = Path("/testbed")
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class SkipTask(Exception):
    pass


def patch_location(patch: str) -> tuple[str, list[int]]:
    paths = [line.split(" b/", 1)[1] for line in patch.splitlines()
             if line.startswith("diff --git a/") and " b/" in line]
    if len(paths) != 1 or not paths[0].endswith(".py"):
        raise SkipTask("not_one_python_file")
    positions = []
    old_line = None
    for line in patch.splitlines():
        match = HUNK.match(line)
        if match:
            old_line = int(match.group(1))
        elif old_line is not None and line.startswith("-") and not line.startswith("---"):
            positions.append(old_line)
            old_line += 1
        elif old_line is not None and line.startswith(" "):
            old_line += 1
    if not positions:
        raise SkipTask("no_removed_source_lines")
    return paths[0], positions


def enclosing_callable(source: str, positions: list[int]) -> tuple[list[str], ast.FunctionDef]:
    tree = ast.parse(source)
    matches = []

    def visit(body: list[ast.stmt], parents: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                if not parents:
                    visit(node.body, [node.name])
            elif isinstance(node, ast.FunctionDef):
                if any(node.lineno <= line <= node.end_lineno for line in positions):
                    matches.append((parents + [node.name], node))

    visit(tree.body, [])
    if not matches or len({tuple(path) for path, _ in matches}) != 1:
        raise SkipTask("not_one_sync_callable")
    return matches[0]


def annotation_kind(node: ast.expr | None, name: str, default: object) -> str:
    annotation = ast.unparse(node).lower() if node is not None else ""
    for kind in ("bytes", "bool", "float", "int", "str", "list", "tuple", "dict"):
        if re.search(rf"\b{kind}\b", annotation):
            return kind
    if default is not None:
        for kind, typ in (("bytes", bytes), ("bool", bool), ("float", float),
                          ("int", int), ("str", str), ("list", list),
                          ("tuple", tuple), ("dict", dict)):
            if isinstance(default, typ):
                return kind
    n = name.lower()
    if n in {"s", "t", "text", "string", "input", "value", "sql", "source", "src",
             "pattern", "name", "prefix", "suffix", "other", "query", "key"} or re.fullmatch(r"(?:s|str|string)\d+", n):
        return "str"
    if n in {"n", "k", "i", "j", "index", "length", "size", "count", "limit"}:
        return "int"
    return ""


def parameters(node: ast.FunctionDef, method: bool) -> list[tuple[str, str]]:
    if node.args.vararg or node.args.kwarg or any(default is None for default in node.args.kw_defaults):
        raise SkipTask("variadic_or_required_keyword_only_signature")
    all_args = node.args.posonlyargs + node.args.args
    # Probe the required surface and let optional parameters use their own
    # defaults. This supports generic calls without inventing complex option
    # objects or task-specific adapters.
    args = all_args[:len(all_args) - len(node.args.defaults)]
    if method:
        if not args or args[0].arg not in {"self", "cls"} or args[0].arg == "cls":
            raise SkipTask("unsupported_method_binding")
        args = args[1:]
    if not 1 <= len(args) <= 4:
        raise SkipTask("unsupported_arity")
    kinds = [(arg.arg, annotation_kind(arg.annotation, arg.arg, None)) for arg in args]
    if any(not kind for _, kind in kinds):
        raise SkipTask("unknown_parameter_type")
    return kinds


def value_bank(kind: str, literals: list[str]) -> list[object]:
    if kind == "str":
        base = ["", "a", "b", "ab", "abc", "A", "0", "1", " ", "\n", "_", "-", "a b", "a_b"]
        base += [s for s in literals if len(s) <= 32]
        return list(dict.fromkeys(base + [f"s{i}" for i in range(300)]))
    if kind == "bytes":
        return [list(s.encode()) for s in value_bank("str", literals)]
    if kind == "int":
        return list(range(-128, 129))
    if kind == "float":
        return [i / 4 for i in range(-128, 129)]
    if kind == "bool":
        return [False, True]
    if kind in {"list", "tuple"}:
        return [[]] + [[i] for i in range(-20, 21)] + [[i, i + 1] for i in range(-128, 129)]
    if kind == "dict":
        return [{}] + [{"x": i} for i in range(-128, 129)] + [{"x": i, "y": i + 1} for i in range(-128, 129)]
    raise SkipTask("unsupported_parameter_type")


def make_bank(task_id: str, source: str, params: list[tuple[str, str]]) -> list[list[object]]:
    literals = sorted({n.value for n in ast.walk(ast.parse(source))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)})[:32]
    pools = [value_bank(kind, literals) for _, kind in params]
    rng = random.Random(int(hashlib.sha256(task_id.encode()).hexdigest(), 16))
    cases = []
    seen = set()
    def add(args: list[object]) -> None:
        key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            cases.append(args)

    # Generic relational fuzzing: independent random strings almost never
    # exercise equality, prefix, overlap, or one-edit-distance branches.
    if len(params) == 2 and params[0][1] == params[1][1] == "str":
        seeds = ["", "a", "ab", "abc", "A", "0", "a b", "a_b"]
        seeds += [s for s in literals if len(s) <= 32]
        seeds += [f"s{i}" for i in range(64)]
        for seed in dict.fromkeys(seeds):
            for pair in ([seed, seed], [seed, seed + "a"],
                         [seed, seed[:-1]], [seed, "a" + seed],
                         [seed, seed.upper()], [seed, seed[::-1]]):
                add(pair)
                if len(cases) >= 128:
                    break
            if len(cases) >= 128:
                break
    for _ in range(20000):
        args = [rng.choice(pool) for pool in pools]
        add(args)
        if len(cases) == 256:
            return cases
    raise SkipTask("fewer_than_256_distinct_inputs")


def observe(payload: dict) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    module = importlib.import_module(payload["module"])
    path = payload["callable"]
    values = []

    def encode(value):
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if isinstance(value, bytes):
            return {"__bytes__": value.hex()}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {k: encode(v) for k, v in value.items()}
        raise SkipTask(f"unsupported_return_type_{type(value).__name__}")

    for args in payload["cases"]:
        try:
            target = getattr(module, path[0])
            if len(path) == 2:
                target = getattr(target(), path[1])
            decoded = [bytes(v) if kind == "bytes" else tuple(v) if kind == "tuple" else v
                       for v, (_, kind) in zip(args, payload["params"])]
            value = target(*decoded)
            values.append({"kind": "return", "value": encode(value)})
        except SkipTask:
            raise
        except (TypeError, ValueError, KeyError, IndexError, AttributeError,
                RuntimeError, ZeroDivisionError, UnicodeError, OverflowError) as exc:
            values.append({"kind": "exception", "type": type(exc).__name__})
        except Exception as exc:
            values.append({"kind": "exception", "type": type(exc).__name__})
    return values


def run_observer(payload: dict) -> list[dict]:
    result = subprocess.run(
        [TESTBED_PYTHON, __file__, "--observe"], cwd=ROOT,
        input=json.dumps(payload, ensure_ascii=False), text=True,
        capture_output=True, timeout=90, check=False,
    )
    if result.returncode != 0:
        raise SkipTask("callable_import_or_execution_failed")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SkipTask("observer_stdout_not_json") from exc


def qualify(task: dict) -> dict:
    path, positions = patch_location(task["patch"])
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise SkipTask("source_path_missing")
    clean_source = target.read_text(encoding="utf-8")
    callable_path, node = enclosing_callable(clean_source, positions)
    if len(callable_path) > 2:
        raise SkipTask("nested_callable")
    params = parameters(node, method=len(callable_path) == 2)
    cases = make_bank(task["instance_id"], clean_source, params)
    module = path.removesuffix(".py").replace("/", ".")
    if module.startswith("src."):
        module = module[4:]
    payload = {"module": module, "callable": callable_path, "params": params, "cases": cases}
    reference = run_observer(payload)
    if reference != run_observer(payload):
        raise SkipTask("reference_nondeterministic")
    injection = subprocess.run(["git", "apply", "-"], cwd=ROOT, input=task["patch"],
                               text=True, capture_output=True, check=False)
    if injection.returncode != 0:
        raise SkipTask("official_patch_did_not_apply")
    reversal = None
    try:
        buggy_source = target.read_text(encoding="utf-8")
        buggy_path, buggy_node = enclosing_callable(buggy_source, positions)
        if buggy_path != callable_path or ast.dump(buggy_node.args) != ast.dump(node.args):
            raise SkipTask("callable_signature_changed")
        buggy = run_observer(payload)
    finally:
        reversal = subprocess.run(["git", "apply", "-R", "-"], cwd=ROOT, input=task["patch"],
                                  text=True, capture_output=True, check=False)
    if reversal.returncode != 0 or target.read_text(encoding="utf-8") != clean_source:
        raise SkipTask("official_patch_reversal_failed")
    if len(reference) != 256 or len(buggy) != 256:
        raise SkipTask("observation_count_mismatch")
    disagreement = sum(a != b for a, b in zip(reference, buggy))
    if not 13 <= disagreement <= 243:
        raise SkipTask("q_outside_0.05_to_0.95_headroom")
    return {"status": "q_valid", "instance_id": task["instance_id"],
            "module": module, "callable": callable_path, "params": params,
            "cases": cases, "reference": reference,
            "q_initial": 1 - disagreement / 256,
            "disagreement_count": disagreement,
            "bank_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


def main() -> None:
    if sys.argv[1:] == ["--observe"]:
        print(json.dumps(observe(json.load(sys.stdin)), ensure_ascii=False))
        return
    task = json.load(sys.stdin)
    try:
        result = qualify(task)
    except (SkipTask, SyntaxError, UnicodeError, OSError, subprocess.TimeoutExpired) as exc:
        result = {"status": "skip", "instance_id": task.get("instance_id"),
                  "reason": str(exc) if isinstance(exc, SkipTask) else type(exc).__name__}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
