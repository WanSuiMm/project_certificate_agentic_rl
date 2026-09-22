"""Verify the immutable copied prior-artifact surface."""

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "baselines" / "prior" / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> list[str]:
    failures: list[str] = []
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        target = PROJECT_ROOT / Path(relative)
        if not target.is_file():
            failures.append(f"line {line_number}: missing {relative}")
            continue
        actual = sha256(target)
        if actual != expected:
            failures.append(
                f"line {line_number}: hash mismatch for {relative}: {actual}"
            )
    return failures


def main() -> int:
    failures = audit()
    if failures:
        print("PRIOR_AUDIT_FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PRIOR_AUDIT_PASS")
    print(f"manifest={MANIFEST.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
