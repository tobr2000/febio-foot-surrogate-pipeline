#!/usr/bin/env python3
"""Estimate archive compression ratios without creating a full archive.

The script samples files across one or more dataset directories, streams their
bytes through ZIP-compatible DEFLATE, and writes a JSON report. It never changes
the source dataset and does not retain compressed payloads.

Example:
    python estimate_dataset_compression.py \
      /path/to/shards/anatomic_v9_contact_v1_modelready \
      /path/to/shards_modelready/anatomic_v10_contact_v1 \
      --sample-gib 5 --output compression_estimate.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zlib
from collections import Counter
from pathlib import Path


def discover(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def stratified_sample(files: list[Path], byte_budget: int, seed: int) -> list[Path]:
    """Sample across file sizes while staying close to the byte budget."""
    if sum(path.stat().st_size for path in files) <= byte_budget:
        return files
    rng = random.Random(seed)
    ordered = sorted(files, key=lambda path: path.stat().st_size)
    buckets = [ordered[i::10] for i in range(10)]
    for bucket in buckets:
        rng.shuffle(bucket)
    selected: list[Path] = []
    total = 0
    cursor = 0
    while total < byte_budget and any(buckets):
        bucket = buckets[cursor % len(buckets)]
        cursor += 1
        if not bucket:
            continue
        path = bucket.pop()
        selected.append(path)
        total += path.stat().st_size
    return selected


def compress_files(files: list[Path], level: int, chunk_size: int) -> tuple[int, int, float]:
    raw = 0
    compressed = 0
    start = time.monotonic()
    for path in files:
        compressor = zlib.compressobj(level=level, wbits=-15)
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                raw += len(chunk)
                compressed += len(compressor.compress(chunk))
        compressed += len(compressor.flush())
    return raw, compressed, time.monotonic() - start


def analyze(root: Path, args: argparse.Namespace) -> dict:
    files = discover(root)
    total_bytes = sum(path.stat().st_size for path in files)
    sample = stratified_sample(files, args.sample_bytes, args.seed)
    sample_raw, sample_compressed, elapsed = compress_files(sample, args.level, args.chunk_mib * 1024**2)
    ratio = sample_compressed / sample_raw if sample_raw else math.nan
    projected = round(total_bytes * ratio) if sample_raw else None
    extensions = Counter((path.suffix.lower() or "[none]") for path in files)
    return {
        "root": str(root.resolve()),
        "files": len(files),
        "total_bytes": total_bytes,
        "sample_files": len(sample),
        "sample_raw_bytes": sample_raw,
        "sample_deflate_bytes": sample_compressed,
        "sample_ratio": ratio,
        "projected_archive_bytes": projected,
        "projected_under_50_gib": bool(projected is not None and projected <= 50 * 1024**3),
        "compression_level": args.level,
        "elapsed_seconds": elapsed,
        "throughput_mib_s": sample_raw / 1024**2 / elapsed if elapsed else None,
        "extensions": dict(extensions.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--sample-gib", type=float, default=5.0)
    parser.add_argument("--level", type=int, choices=range(0, 10), default=6)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, default=Path("compression_estimate.json"))
    args = parser.parse_args()
    args.sample_bytes = round(args.sample_gib * 1024**3)
    missing = [str(root) for root in args.roots if not root.is_dir()]
    if missing:
        parser.error(f"not directories: {', '.join(missing)}")
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "stratified file sample, raw DEFLATE stream per file (ZIP-compatible)",
        "datasets": [analyze(root, args) for root in args.roots],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
