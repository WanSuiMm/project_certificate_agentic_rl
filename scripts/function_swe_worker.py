"""Single-request scorer. Run only inside a genuine code-execution sandbox.

stdin: {"task": ..., "source": ..., "terminal": bool}
stdout: one JSON score object. Never place secrets in this worker's environment.
"""

from __future__ import annotations

import json
import sys

from function_swe import score_source


def main() -> None:
    request = json.load(sys.stdin)
    if not isinstance(request.get("terminal"), bool):
        raise ValueError("terminal must be bool")
    if not isinstance(request.get("include_proxy", True), bool):
        raise ValueError("include_proxy must be bool")
    score = score_source(request["task"], request["source"],
                         terminal=request["terminal"],
                         include_proxy=request.get("include_proxy", True))
    print(json.dumps(score, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
