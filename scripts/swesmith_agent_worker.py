"""Official-image SWE-smith initialization and candidate scoring worker.

Run only in a fresh, network-blocked Modal sandbox. Model code is written into
the isolated checkout and never executed on the trainer machine.
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


def run(request: dict) -> dict:
    task = request["task"]
    path, _ = patch_location(task["patch"])
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise ValueError("invalid target path")
    clean_source = target.read_text(encoding="utf-8")
    selectors = list(dict.fromkeys(task["FAIL_TO_PASS"] + task["PASS_TO_PASS"]))
    clean_test = tests(selectors) if request["mode"] == "init" else None
    injection = subprocess.run(["git", "apply", "-"], cwd=ROOT, input=task["patch"],
                               text=True, capture_output=True, check=False)
    if injection.returncode:
        raise RuntimeError("official_bug_injection_failed")
    buggy_source = target.read_text(encoding="utf-8")
    if request["mode"] == "init":
        buggy_test = tests(task["FAIL_TO_PASS"])
        return {"status": "initialized", "instance_id": task["instance_id"],
                "target_path": path, "clean_source_sha256": hashlib.sha256(clean_source.encode()).hexdigest(),
                "buggy_source": buggy_source,
                "buggy_source_sha256": hashlib.sha256(buggy_source.encode()).hexdigest(),
                "clean_tests": clean_test, "buggy_f2p": buggy_test,
                "gold_self_consistent": bool(clean_test["valid"] and clean_test["exitcode"] == 0
                                              and buggy_test["valid"] and buggy_test["exitcode"] != 0)}
    if request["mode"] not in {"score", "proxy_only"}:
        raise ValueError("unknown mode")
    if hashlib.sha256(buggy_source.encode()).hexdigest() != request["buggy_source_sha256"]:
        raise RuntimeError("buggy_source_hash_mismatch")
    candidate = request["source"]
    if len(candidate) > 300_000:
        raise ValueError("candidate_source_too_large")
    target.write_text(candidate, encoding="utf-8")
    if request["mode"] == "proxy_only":
        bank = request["q_bank"]
        reference = bank["reference"]
        if len(reference) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        try:
            observed = run_observer({key: bank[key] for key in ("module", "callable", "params", "cases")})
        except (SkipTask, subprocess.TimeoutExpired) as exc:
            return {"status": "q_invalid_candidate", "instance_id": task["instance_id"],
                    "reason": type(exc).__name__, "q": None}
        if len(observed) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        return {"status": "q_scored", "instance_id": task["instance_id"],
                "q": sum(a == b for a, b in zip(observed, reference)) / 256}
    public = tests(selectors)
    if not public["valid"]:
        failed = {"status": "invalid_candidate", "instance_id": task["instance_id"],
                  "public": public, "p_T": 0.0, "solved": False}
        if request.get("include_proxy", False):
            failed["q"] = 0.0
        return failed
    result = {"status": "scored", "instance_id": task["instance_id"],
              "public": public, "p_T": public.get("pass_fraction", 0.0) if public["valid"] else 0.0,
              "solved": bool(public["valid"] and public["exitcode"] == 0
                             and public["passed"] == public["total"])}
    if request.get("include_proxy", False):
        bank = request["q_bank"]
        reference = bank["reference"]
        if len(reference) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        try:
            observed = run_observer({key: bank[key] for key in ("module", "callable", "params", "cases")})
        except (SkipTask, subprocess.TimeoutExpired) as exc:
            observed = []
            result["q_invalid_candidate"] = type(exc).__name__
        if observed and len(observed) != 256:
            raise RuntimeError("q_bank_observation_count_mismatch")
        result["q"] = sum(a == b for a, b in zip(observed, reference)) / 256
    return result


def main() -> None:
    try:
        result = run(json.load(sys.stdin))
    except Exception as exc:
        result = {"status": "worker_error", "reason": type(exc).__name__, "detail": str(exc)[:500]}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
