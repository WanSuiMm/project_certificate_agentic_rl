"""Official-image SWE-smith initialization and candidate scoring worker.

Run only in a network-blocked Modal sandbox. ``--serve`` keeps one injected
task checkout alive for many public tests or offline q observations. Model
code is never executed on the trainer machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from swesmith_q_qualifier_worker import ROOT, TESTBED_PYTHON, SkipTask, patch_location, run_observer


PYTEST_DRIVER = r'''
import json, pytest, sys
class Results:
    def __init__(self): self.outcomes = {}
    def pytest_runtest_logreport(self, report):
        if report.when == "call": self.outcomes[report.nodeid] = report.outcome
        elif report.when == "setup" and report.failed: self.outcomes[report.nodeid] = "failed"
results = Results()
code = pytest.main(["-q", "--disable-warnings", "--tb=no", *json.loads(sys.stdin.read())], plugins=[results])
print("SWE_SMITH_RESULT=" + json.dumps({"exitcode": int(code), "outcomes": results.outcomes}))
'''


def tests(selectors: list[str]) -> dict:
    if not selectors:
        return {"valid": False, "reason": "no_test_selectors"}
    try:
        run = subprocess.run([TESTBED_PYTHON, "-c", PYTEST_DRIVER], cwd=ROOT,
                             input=json.dumps(selectors), text=True, capture_output=True,
                             timeout=240, check=False)
    except subprocess.TimeoutExpired:
        return {"valid": False, "reason": "pytest_timeout"}
    marker = "SWE_SMITH_RESULT="
    lines = [line.split(marker, 1)[1] for line in run.stdout.splitlines() if marker in line]
    if run.returncode or len(lines) != 1:
        return {"valid": False, "reason": "pytest_driver_failure",
                "returncode": run.returncode, "stderr_tail": run.stderr[-500:]}
    result = json.loads(lines[0])
    outcomes = result["outcomes"]
    passed = sum(outcome == "passed" for outcome in outcomes.values())
    total = len(outcomes)
    return {"valid": total >= len(selectors) and result["exitcode"] in (0, 1),
            "exitcode": result["exitcode"], "passed": passed, "total": total,
            "pass_fraction": passed / total if total else 0.0,
            "output_tail": run.stdout[-1000:]}


class TaskWorker:
    def __init__(self) -> None:
        self.task = None
        self.target = None
        self.selectors = None
        self.buggy_source_sha256 = None
        self.other_status = None
        self.q_bank = None

    def non_target_status(self) -> tuple[str, ...]:
        """Detect tracked/untracked test mutations that could contaminate later states."""
        result = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=normal"],
                                cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("git_status_failed")
        target_path = self.target.relative_to(ROOT).as_posix()
        return tuple(sorted(line for line in result.stdout.splitlines()
                            if line[3:] != target_path))

    def initialize(self, task: dict, mode: str) -> dict:
        if self.task is not None or mode not in {"init", "source_only"}:
            raise RuntimeError("task already initialized or bad init mode")
        path, _ = patch_location(task["patch"])
        target = (ROOT / path).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError("invalid target path")
        clean_source = target.read_text(encoding="utf-8")
        selectors = list(dict.fromkeys(task["FAIL_TO_PASS"] + task["PASS_TO_PASS"]))
        clean_test = tests(selectors) if mode == "init" else None
        injection = subprocess.run(["git", "apply", "-"], cwd=ROOT, input=task["patch"],
                                   text=True, capture_output=True, check=False)
        if injection.returncode:
            raise RuntimeError("official_bug_injection_failed")
        buggy_source = target.read_text(encoding="utf-8")
        buggy_hash = hashlib.sha256(buggy_source.encode()).hexdigest()
        self.task, self.target = task, target
        self.selectors, self.buggy_source_sha256 = selectors, buggy_hash
        common = {"instance_id": task["instance_id"], "target_path": path,
                  "buggy_source": buggy_source, "buggy_source_sha256": buggy_hash}
        if mode == "source_only":
            self.other_status = self.non_target_status()
            return {"status": "source_only", **common}
        buggy_test = tests(task["FAIL_TO_PASS"])
        self.other_status = self.non_target_status()
        return {"status": "initialized", **common,
                "clean_source_sha256": hashlib.sha256(clean_source.encode()).hexdigest(),
                "clean_tests": clean_test, "buggy_f2p": buggy_test,
                "gold_self_consistent": bool(clean_test["valid"] and clean_test["exitcode"] == 0
                                              and buggy_test["valid"] and buggy_test["exitcode"] != 0)}

    def observe_q(self, bank: dict) -> dict:
        reference = bank["reference"]
        if len(reference) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        try:
            observed = run_observer({key: bank[key] for key in ("module", "callable", "params", "cases")})
        except (SkipTask, subprocess.TimeoutExpired) as exc:
            return {"status": "q_invalid_candidate", "instance_id": self.task["instance_id"],
                    "reason": type(exc).__name__, "q": None}
        if len(observed) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        return {"status": "q_scored", "instance_id": self.task["instance_id"],
                "q": sum(a == b for a, b in zip(observed, reference)) / 256}

    def set_q_bank(self, bank: dict) -> dict:
        if self.task is None or self.q_bank is not None or len(bank["reference"]) != 256:
            raise RuntimeError("task not initialized, q bank repeated, or wrong bank size")
        self.q_bank = bank
        return {"status": "q_bank_ready", "instance_id": self.task["instance_id"]}

    def score(self, request: dict) -> dict:
        if self.task is None or request["mode"] not in {"score", "proxy_only"}:
            raise RuntimeError("task not initialized or bad scoring mode")
        if (request.get("instance_id", self.task["instance_id"]) != self.task["instance_id"]
                or request["buggy_source_sha256"] != self.buggy_source_sha256):
            raise RuntimeError("task_or_buggy_source_hash_mismatch")
        candidate = request["source"]
        if len(candidate) > 300_000:
            raise ValueError("candidate_source_too_large")
        self.target.write_text(candidate, encoding="utf-8")
        if request["mode"] == "proxy_only":
            bank = request.get("q_bank") or self.q_bank
            if bank is None:
                raise RuntimeError("q_bank_missing")
            result = self.observe_q(bank)
            if self.non_target_status() != self.other_status:
                raise RuntimeError("observer_mutated_other_repository_files")
            return result
        public = tests(self.selectors)
        if self.non_target_status() != self.other_status:
            raise RuntimeError("public_tests_mutated_other_repository_files")
        if not public["valid"]:
            failed = {"status": "invalid_candidate", "instance_id": self.task["instance_id"],
                      "public": public, "p_T": 0.0, "solved": False}
            if request.get("include_proxy", False):
                failed["q"] = 0.0
            return failed
        result = {"status": "scored", "instance_id": self.task["instance_id"],
                  "public": public, "p_T": public["pass_fraction"],
                  "solved": bool(public["exitcode"] == 0 and public["passed"] == public["total"])}
        if request.get("include_proxy", False):
            bank = request.get("q_bank") or self.q_bank
            if bank is None:
                raise RuntimeError("q_bank_missing")
            q = self.observe_q(bank)
            result["q"] = q["q"] if q["q"] is not None else 0.0
            if q["status"] != "q_scored":
                result["q_invalid_candidate"] = q["reason"]
        return result


def run(request: dict) -> dict:
    worker = TaskWorker()
    mode = request["mode"]
    if mode in {"init", "source_only"}:
        return worker.initialize(request["task"], mode)
    worker.initialize(request["task"], "source_only")
    return worker.score(request)


def serve() -> None:
    worker = TaskWorker()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            mode = request["mode"]
            if mode in {"init", "source_only"}:
                result = worker.initialize(request["task"], mode)
            elif mode == "set_q_bank":
                result = worker.set_q_bank(request["q_bank"])
            else:
                result = worker.score(request)
        except Exception as exc:
            result = {"status": "worker_error", "reason": type(exc).__name__,
                      "detail": str(exc)[:500]}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    try:
        result = run(json.load(sys.stdin))
    except Exception as exc:
        result = {"status": "worker_error", "reason": type(exc).__name__, "detail": str(exc)[:500]}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--serve":
        serve()
    else:
        main()
