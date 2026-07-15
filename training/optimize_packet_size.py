from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleRuntime:
    sample_id: int
    duration_sec: float
    normal_termination: bool
    source: str


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def rows_from_batch_json(path: Path) -> list[SampleRuntime]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []
    raw_rows = data.get("samples")
    if raw_rows is None:
        raw_rows = data.get("summaries")
    if not isinstance(raw_rows, list):
        return []

    rows: list[SampleRuntime] = []
    for idx, row in enumerate(raw_rows):
        if not isinstance(row, dict) or "duration_sec" not in row:
            continue
        try:
            sample_id = int(row.get("sample_id", idx))
            duration_sec = float(row["duration_sec"])
        except Exception:
            continue
        rows.append(
            SampleRuntime(
                sample_id=sample_id,
                duration_sec=duration_sec,
                normal_termination=bool(row.get("normal_termination", False)),
                source=str(path),
            )
        )
    return rows


def rows_from_json_lines(path: Path) -> list[SampleRuntime]:
    rows: list[SampleRuntime] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line.startswith("{") or "duration_sec" not in line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if not isinstance(data, dict) or "sample_id" not in data:
            continue
        try:
            rows.append(
                SampleRuntime(
                    sample_id=int(data.get("sample_id", idx)),
                    duration_sec=float(data["duration_sec"]),
                    normal_termination=bool(data.get("normal_termination", False)),
                    source=str(path),
                )
            )
        except Exception:
            continue
    return rows


def iter_candidate_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from path.rglob("batch_*.json")
            yield from path.rglob("*.out")
            yield from path.rglob("*.log")
        elif path.exists():
            yield path


def load_runtimes(paths: list[Path]) -> list[SampleRuntime]:
    by_sample: dict[int, SampleRuntime] = {}
    for path in iter_candidate_files(paths):
        rows: list[SampleRuntime] = []
        if path.suffix.lower() == ".json" and path.name.startswith("batch_"):
            rows = rows_from_batch_json(path)
        if not rows and path.suffix.lower() in {".out", ".log", ".txt"}:
            rows = rows_from_json_lines(path)
        for row in rows:
            # Keep the longest record for duplicate sample ids. It usually comes
            # from the real solve rather than a partial/retried parse artifact.
            previous = by_sample.get(row.sample_id)
            if previous is None or row.duration_sec > previous.duration_sec:
                by_sample[row.sample_id] = row
    return [by_sample[key] for key in sorted(by_sample)]


def expand_sequence(rows: list[SampleRuntime], target_attempts: int) -> list[SampleRuntime]:
    if not rows:
        return []
    if target_attempts <= len(rows):
        return rows[:target_attempts]
    repeats = math.ceil(target_attempts / len(rows))
    expanded: list[SampleRuntime] = []
    for _ in range(repeats):
        expanded.extend(rows)
    return expanded[:target_attempts]


def simulate_batch(rows: list[SampleRuntime], workers: int, overhead_sec: float) -> tuple[float, float, int]:
    loads = [0.0] * max(1, workers)
    total_work = 0.0
    successes = 0
    for row in rows:
        worker_idx = min(range(len(loads)), key=loads.__getitem__)
        loads[worker_idx] += row.duration_sec
        total_work += row.duration_sec
        successes += int(row.normal_termination)
    wall = (max(loads) if loads else 0.0) + overhead_sec
    idle_worker_sec = max(0.0, wall * len(loads) - total_work)
    return wall, idle_worker_sec, successes


def schedule_batches(batch_walls: list[float], concurrent_jobs: int) -> float:
    slots = [0.0] * max(1, concurrent_jobs)
    for wall in batch_walls:
        slot_idx = min(range(len(slots)), key=slots.__getitem__)
        slots[slot_idx] += wall
    return max(slots) if slots else 0.0


def simulate_dataset(
    rows: list[SampleRuntime],
    packet_size: int,
    workers: int,
    concurrent_jobs: int,
    overhead_sec: float,
) -> dict[str, float]:
    batch_walls: list[float] = []
    idle_worker_sec = 0.0
    successes = 0
    for start in range(0, len(rows), packet_size):
        batch = rows[start : start + packet_size]
        wall, idle, ok = simulate_batch(batch, workers=workers, overhead_sec=overhead_sec)
        batch_walls.append(wall)
        idle_worker_sec += idle
        successes += ok

    serial_wall = sum(batch_walls)
    makespan = schedule_batches(batch_walls, concurrent_jobs=concurrent_jobs)
    total_work = sum(row.duration_sec for row in rows)
    worker_capacity = sum(wall * workers for wall in batch_walls)
    utilization = (total_work / worker_capacity) if worker_capacity > 0 else 0.0
    hours = makespan / 3600.0 if makespan > 0 else 0.0
    return {
        "packet_size": float(packet_size),
        "batches": float(len(batch_walls)),
        "attempted": float(len(rows)),
        "successful": float(successes),
        "success_fraction": float(successes / len(rows)) if rows else 0.0,
        "makespan_hours": hours,
        "serial_batch_hours": serial_wall / 3600.0,
        "mean_batch_hours": (statistics.mean(batch_walls) / 3600.0) if batch_walls else 0.0,
        "max_batch_hours": (max(batch_walls) / 3600.0) if batch_walls else 0.0,
        "worker_utilization": utilization,
        "idle_worker_hours": idle_worker_sec / 3600.0,
        "attempts_per_hour": (len(rows) / hours) if hours > 0 else 0.0,
        "successes_per_hour": (successes / hours) if hours > 0 else 0.0,
    }


def summarize_bootstrap(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = values[0].keys()
    summary: dict[str, float] = {}
    for key in keys:
        series = [row[key] for row in values]
        summary[key] = statistics.mean(series)
        if key in {"makespan_hours", "successes_per_hour", "worker_utilization"}:
            ordered = sorted(series)
            summary[f"{key}_p10"] = ordered[max(0, int(0.10 * (len(ordered) - 1)))]
            summary[f"{key}_p50"] = ordered[max(0, int(0.50 * (len(ordered) - 1)))]
            summary[f"{key}_p90"] = ordered[max(0, int(0.90 * (len(ordered) - 1)))]
    return summary


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_packet_sizes(
    inputs: list[Path],
    workers: int,
    array_concurrency: int,
    min_packet: int,
    max_packet: int | None,
    target_attempts: int | None = None,
    target_successes: int | None = None,
    bootstrap: int = 500,
    overhead_sec: float = 0.0,
    seed: int = 13,
    out_dir: Path | None = None,
) -> dict[str, object]:
    rows = load_runtimes(inputs)
    if not rows:
        raise ValueError("No per-sample durations found in the provided inputs.")

    success_fraction = sum(row.normal_termination for row in rows) / len(rows)
    effective_target_attempts = target_attempts
    if effective_target_attempts is None and target_successes is not None:
        effective_target_attempts = math.ceil(target_successes / max(success_fraction, 1e-9))
    if effective_target_attempts is None:
        effective_target_attempts = len(rows)

    effective_max_packet = max_packet or min(max(workers * 4, min_packet), effective_target_attempts)
    packet_sizes = list(range(min_packet, effective_max_packet + 1))
    observed_rows = expand_sequence(rows, effective_target_attempts)

    deterministic = [
        simulate_dataset(
            observed_rows,
            packet_size=packet_size,
            workers=workers,
            concurrent_jobs=array_concurrency,
            overhead_sec=overhead_sec,
        )
        for packet_size in packet_sizes
    ]

    rng = random.Random(seed)
    bootstrap_rows: list[dict[str, float]] = []
    if bootstrap > 0:
        for packet_size in packet_sizes:
            reps = []
            for _ in range(bootstrap):
                sampled = [rng.choice(rows) for _ in range(effective_target_attempts)]
                reps.append(
                    simulate_dataset(
                        sampled,
                        packet_size=packet_size,
                        workers=workers,
                        concurrent_jobs=array_concurrency,
                        overhead_sec=overhead_sec,
                    )
                )
            bootstrap_rows.append(summarize_bootstrap(reps))

    best_observed = max(deterministic, key=lambda row: (row["successes_per_hour"], row["worker_utilization"]))
    best_bootstrap = (
        max(bootstrap_rows, key=lambda row: (row["successes_per_hour"], row["worker_utilization"]))
        if bootstrap_rows
        else None
    )

    summary: dict[str, object] = {
        "inputs": [str(path) for path in inputs],
        "observed_samples": len(rows),
        "observed_successes": int(sum(row.normal_termination for row in rows)),
        "observed_success_fraction": success_fraction,
        "duration_sec_min": min(row.duration_sec for row in rows),
        "duration_sec_mean": statistics.mean(row.duration_sec for row in rows),
        "duration_sec_median": statistics.median(row.duration_sec for row in rows),
        "duration_sec_max": max(row.duration_sec for row in rows),
        "workers": workers,
        "array_concurrency": array_concurrency,
        "target_attempts": effective_target_attempts,
        "target_successes": target_successes,
        "overhead_sec": overhead_sec,
        "best_observed_order": best_observed,
        "best_bootstrap_mean": best_bootstrap,
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "packet_size_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_csv(out_dir / "packet_size_observed_order.csv", deterministic)
        if bootstrap_rows:
            write_csv(out_dir / "packet_size_bootstrap.csv", bootstrap_rows)

    return {
        "summary": summary,
        "observed_order": deterministic,
        "bootstrap": bootstrap_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate efficient FEBio packet sizes from observed per-sample runtimes. "
            "Reads batch JSON files or JSON lines in SLURM .out/.log files."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Batch JSON files, SLURM logs, or directories.")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--array-concurrency", type=int, default=1)
    parser.add_argument("--min-packet", type=int, default=1)
    parser.add_argument("--max-packet", type=int, default=None)
    parser.add_argument("--target-attempts", type=int, default=None)
    parser.add_argument("--target-successes", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--overhead-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_packet_sizes(
        inputs=args.inputs,
        workers=args.workers,
        array_concurrency=args.array_concurrency,
        min_packet=args.min_packet,
        max_packet=args.max_packet,
        target_attempts=args.target_attempts,
        target_successes=args.target_successes,
        bootstrap=args.bootstrap,
        overhead_sec=args.overhead_sec,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    summary = result["summary"]

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
