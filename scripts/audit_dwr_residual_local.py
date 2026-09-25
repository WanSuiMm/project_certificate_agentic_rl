#!/usr/bin/env python3
"""Locally remeasure frozen Oracle Credit P1/P2 per-probe agreement.

Run this script with WSL Python. Saved candidate code executes only in a
networkless bubblewrap namespace containing the pinned public source checkout,
the frozen candidate module, and explicit Python dependencies. No model, Modal,
public tests, or new trajectories are run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_PARENTS = Path(__file__).resolve().parents
PROJECT = _PARENTS[1] if len(_PARENTS) > 1 else Path("/")
INPUTS = PROJECT / "runs/oracle_credit_6x16x4_v01/frozen_inputs_v01"
SOURCE = PROJECT / "runs/oracle_credit_6x16x4_v01/run_v03"
P2_Q = PROJECT / "evidence/oracle_credit_p2_q_v01/observations.jsonl"
DEFAULT_OUTPUT = PROJECT / "runs/dwr_residual_audit_v01"
BASE_COMMIT = "c4a72f59aafe8db42c4015709078064535dc4191"


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def observe(payload: dict) -> list[dict]:
    """Mirror the frozen qualifier's observer, including exception encoding."""
    sys.path.insert(0, "/testbed")
    module = importlib.import_module(payload["module"])

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
        raise TypeError(f"unsupported_return_type_{type(value).__name__}")

    values = []
    for args in payload["cases"]:
        try:
            target = getattr(module, payload["callable"][0])
            if len(payload["callable"]) == 2:
                target = getattr(target(), payload["callable"][1])
            decoded = [bytes(v) if kind == "bytes" else tuple(v) if kind == "tuple" else v
                       for v, (_, kind) in zip(args, payload["params"])]
            values.append({"kind": "return", "value": encode(target(*decoded))})
        except Exception as exc:
            values.append({"kind": "exception", "type": type(exc).__name__})
    return values


def check_setup(repo: Path, deps: Path) -> None:
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != BASE_COMMIT:
        raise ValueError("support repository is not at the frozen base commit")
    if not (repo / "string2string/misc/basic_functions.py").is_file():
        raise ValueError("missing exact support source")
    if not (deps / "numpy").is_dir() or not (deps / "joblib").is_dir() or not (deps / "tqdm").is_dir():
        raise ValueError("missing local numpy/joblib/tqdm dependencies")
    if shutil.which("bwrap") is None:
        raise ValueError("bubblewrap is required to isolate saved candidate code")


def score_source(bank: dict, source: str, repo: Path, deps: Path, scratch: Path) -> str:
    module_file = scratch / Path(*bank["module"].split(".")).with_suffix(".py")
    if not module_file.is_file():
        raise ValueError(f"missing module in pinned source: {module_file}")
    module_file.write_text(source, encoding="utf-8")
    payload = {key: bank[key] for key in ("module", "callable", "params", "cases")}
    command = [
        "bwrap", "--die-with-parent", "--unshare-net", "--unshare-pid",
        "--clearenv", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--dir", "/etc", "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        "--dev", "/dev", "--proc", "/proc",
        "--tmpfs", "/tmp", "--dir", "/opt", "--ro-bind", str(scratch), "/testbed",
        "--ro-bind", str(deps), "/opt/deps", "--ro-bind", str(Path(__file__).resolve()), "/observer.py",
        "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONPATH", "/testbed:/opt/deps", "--chdir", "/testbed",
        "--", "/usr/bin/python3", "/observer.py", "--observe",
    ]
    result = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False), text=True,
                            capture_output=True, timeout=100, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"isolated observer failed: {result.stderr[-1200:]}")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"observer stdout is not JSON: {result.stdout[-500:]}") from exc
    if len(values) != 256:
        raise ValueError(f"observer returned {len(values)} values, expected 256")
    return "".join("1" if a == b else "0" for a, b in zip(values, bank["reference"]))


def collect(args: argparse.Namespace) -> None:
    repo, deps, output = args.support_repo.resolve(), args.deps.resolve(), args.output.resolve()
    check_setup(repo, deps)
    output.mkdir(parents=True, exist_ok=True)
    banks = {x["instance_id"]: x for x in read_rows(INPUTS / "results.jsonl") if x["status"] == "q_valid"}
    selected = [x for x in json.loads((SOURCE / "run.json").read_text(encoding="utf-8"))["task_ids"]
                if x.startswith("stanfordnlp__string2string.")]
    expected_p2 = {(x["instance_id"], x["sample"], x["replicate"]): x
                   for x in read_rows(P2_Q)}
    jobs = {}
    for row in read_rows(SOURCE / "candidates.jsonl"):
        task = row["instance_id"]
        if task in selected and row["q"] is not None:
            jobs.setdefault((task, row["source_sha256"]), (row["source"], row["q"]))
    for row in read_rows(SOURCE / "states.jsonl"):
        task = row["instance_id"]
        if task not in selected or row["step"] != 2:
            continue
        qrow = expected_p2[(task, row["sample"], row["replicate"])]
        if qrow["q2"] is not None:
            jobs.setdefault((task, row["source_sha256"]), (row["source"], qrow["q2"]))
    if args.limit:
        jobs = dict(list(jobs.items())[:args.limit])
    existing = {(x["instance_id"], x["source_sha256"]): x
                for x in read_rows(output / "vectors.jsonl")} if (output / "vectors.jsonl").exists() else {}
    completed = 0
    with tempfile.TemporaryDirectory(prefix="dwr_observer_") as tmp:
        scratch = Path(tmp) / "testbed"
        shutil.copytree(repo / "string2string", scratch / "string2string")
        # The package initializer eagerly imports unrelated embedding models
        # (and torch). The selected target modules import only the two exact
        # submodules below; defer those unrelated exports in this local mirror.
        (scratch / "string2string/misc/__init__.py").write_text("", encoding="utf-8")
        (scratch / "string2string/search/__init__.py").write_text("", encoding="utf-8")
        with (output / "vectors.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            for (task, sha), (source, expected_q) in jobs.items():
                if digest(source) != sha:
                    raise ValueError(f"source hash mismatch: {task} {sha}")
                if (task, sha) in existing:
                    continue
                bits = score_source(banks[task], source, repo, deps, scratch)
                q = bits.count("1") / 256
                row = {"instance_id": task, "source_sha256": sha, "bank_sha256": banks[task]["bank_sha256"],
                       "match_bits": bits, "q_local": q, "q_frozen": expected_q,
                       "parity": q == expected_q}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                completed += 1
                if completed % 10 == 0 or not row["parity"]:
                    print(f"measured={completed} task={task.rsplit('__', 1)[-1]} parity={row['parity']}", flush=True)
                if not row["parity"]:
                    raise RuntimeError(f"q parity failed: {task} {sha} local={q} frozen={expected_q}")
    print(f"completed={completed} cached={len(existing)} output={output / 'vectors.jsonl'}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--observe":
        print(json.dumps(observe(json.load(sys.stdin)), ensure_ascii=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-repo", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
