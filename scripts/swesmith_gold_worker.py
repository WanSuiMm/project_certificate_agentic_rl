"""Run one official SWE-smith bug-injection self-consistency check in a sandbox."""

import json
import subprocess
import sys


ROOT = "/testbed"


def run(command, *, input_text=None, timeout=120):
    result = subprocess.run(
        command, cwd=ROOT, input=input_text, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-1000:],
    }


def main():
    task = json.load(sys.stdin)
    test = task["FAIL_TO_PASS"][0]
    # SWE-smith installs repository/test dependencies in its `testbed` conda env;
    # the image's default `python` is the bare base interpreter.
    command = ["/opt/miniconda3/envs/testbed/bin/python", "-m", "pytest", "-q",
               "--disable-warnings", "--maxfail=1", test]
    clean_status = run(["git", "status", "--porcelain"])
    if clean_status["returncode"] != 0 or clean_status["stdout_tail"]:
        raise RuntimeError("image is not a clean checkout")
    baseline = run(command)
    injection = run(["git", "apply", "-"], input_text=task["patch"])
    if injection["returncode"] != 0:
        raise RuntimeError(f"official patch did not apply: {injection}")
    buggy = run(command)
    undo = run(["git", "apply", "-R", "-"], input_text=task["patch"])
    if undo["returncode"] != 0:
        raise RuntimeError(f"official patch did not reverse: {undo}")
    restored = run(command)
    result = {
        "instance_id": task["instance_id"],
        "test": test,
        "baseline": baseline,
        "buggy": buggy,
        "restored": restored,
        "self_consistent": baseline["returncode"] == 0
        and buggy["returncode"] != 0 and restored["returncode"] == 0,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
