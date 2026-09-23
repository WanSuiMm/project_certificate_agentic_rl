#!/usr/bin/env python3
"""Parallel HTTP range download, with exact byte-count and whole-file SHA-256.

Designed for a separately verified mirror of a pinned public model blob.
Segments and the assembled file stay outside the repository.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time


def fetch(url: str, root: Path, index: int, start: int, stop: int) -> dict:
    segment = root / f"{index:06d}.part"
    expected = stop - start + 1
    if segment.exists() and segment.stat().st_size == expected:
        return {"index": index, "bytes": expected, "cached": True}
    for attempt in range(1, 5):
        temporary = root / f"{index:06d}.downloading"
        result = subprocess.run(
            ["curl", "--http1.1", "--location", "--range", f"{start}-{stop}",
             "--max-time", "120", "--silent", "--show-error", "--output", str(temporary),
             "--write-out", "%{http_code} %{size_download}", url],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == f"206 {expected}" \
                and temporary.is_file() and temporary.stat().st_size == expected:
            temporary.replace(segment)
            return {"index": index, "bytes": expected, "attempt": attempt}
        time.sleep(attempt)
    raise RuntimeError(f"segment {index} failed after four attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.size < 1 or len(args.sha256) != 64:
        raise ValueError("invalid download specification")
    if args.output.exists():
        raise FileExistsError(args.output)
    chunk = args.chunk_mib * 1024 * 1024
    if chunk < 1024 * 1024:
        raise ValueError("chunk too small")
    root = args.output.with_name(args.output.name + ".segments")
    root.mkdir(parents=True, exist_ok=True)
    ranges = [(i, start, min(args.size - 1, start + chunk - 1))
              for i, start in enumerate(range(0, args.size, chunk))]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, args.url, root, i, start, stop) for i, start, stop in ranges]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 16 == 0 or completed == len(ranges):
                print(json.dumps({"segments": completed, "total": len(ranges),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    assembled = args.output.with_name(args.output.name + ".assembling")
    digest = hashlib.sha256()
    with assembled.open("wb") as destination:
        for index, _, _ in ranges:
            with (root / f"{index:06d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    destination.write(block)
    actual = digest.hexdigest()
    if assembled.stat().st_size != args.size or actual != args.sha256:
        raise RuntimeError(f"whole-file verification failed: size={assembled.stat().st_size} sha256={actual}")
    assembled.replace(args.output)
    print(json.dumps({"status": "verified", "path": str(args.output),
                      "bytes": args.size, "sha256": actual,
                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
