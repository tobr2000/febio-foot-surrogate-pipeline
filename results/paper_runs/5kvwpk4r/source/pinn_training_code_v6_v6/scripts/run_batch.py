from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (
    DEFAULT_RUNS,
    DEFAULT_TEMPLATE,
    ROOT,
    febio_executable,
    load_manifest,
    normal_termination,
    render_feb,
    run_febio,
    write_json,
)
from extract_outputs import extract
from pack_batch import pack_batch


def solver_failure_excerpt(log_text: str, max_lines: int = 30) -> list[str]:
    """Return a compact FEBio failure excerpt suitable for SLURM stdout."""
    lines = [line.rstrip() for line in log_text.splitlines()]
    interesting_terms = (
        "error",
        "fatal",
        "failed",
        "negative jacobian",
        "nan",
        "singular",
        "zero pivot",
        "invalid",
        "exception",
        "abort",
    )
    interesting = [
        line
        for line in lines
        if any(term in line.lower() for term in interesting_terms)
    ]
    excerpt: list[str] = []
    if interesting:
        excerpt.extend(interesting[:max_lines])
    else:
        excerpt.extend(lines[-max_lines:])
    return excerpt[-max_lines:]


def run_one(sample, args, febio: str) -> dict:
    start_time = time.perf_counter()
    run_dir = args.runs / sample.name
    run_dir.mkdir(parents=True, exist_ok=True)
    feb_path = run_dir / f"{sample.name}.feb"
    template, include_base_profile = resolve_template(args.template, sample)
    render_feb(
        template,
        sample,
        feb_path,
        include_base_profile=include_base_profile,
        time_steps=args.time_steps,
        step_size=args.step_size,
    )
    write_json(
        run_dir / "params.json",
        {
            "dataset_id": args.dataset_id or sample.dataset_id,
            "sample_id": sample.sample_id,
            "sample_name": sample.name,
            "params": sample.params,
        },
    )

    result = run_febio(febio, feb_path, run_dir)
    duration_sec = time.perf_counter() - start_time
    solver_log = run_dir / f"{sample.name}_solver_stdout.log"
    solver_log.write_text(result.stdout, encoding="utf-8", errors="ignore")
    ok = result.returncode == 0 and normal_termination(result.stdout)

    sample_summary = {
        "sample_id": sample.sample_id,
        "name": sample.name,
        "dataset_id": args.dataset_id or sample.dataset_id,
        "returncode": result.returncode,
        "normal_termination": ok,
        "run_dir": str(run_dir),
        "template": str(template),
        "duration_sec": duration_sec,
    }

    if ok:
        try:
            sample_summary.update(extract(run_dir, sample.name, include_history=args.include_history))
        except Exception as exc:
            sample_summary["extract_error"] = repr(exc)
    else:
        sample_summary["solver_log"] = str(solver_log)
        sample_summary["solver_error_excerpt"] = solver_failure_excerpt(result.stdout)

    if ok and not args.keep_xplt:
        for xplt in run_dir.glob("*.xplt"):
            xplt.unlink()

    write_json(run_dir / "summary.json", sample_summary)
    return sample_summary


def resolve_template(template_arg: Path, sample) -> tuple[Path, bool]:
    """Return the FEB template and whether base deformation still needs applying."""
    if not template_arg.is_dir():
        return template_arg, True

    base_model_id = int(round(float(sample.params.get("base_model_id", 0.0))))
    profile_path = template_arg / "base_model_profiles.json"
    candidates = [
        template_arg / f"base_{base_model_id:02d}.feb",
        template_arg / f"simplefoot_base_{base_model_id:02d}.feb",
    ]
    if profile_path.exists():
        try:
            profiles = json.loads(profile_path.read_text(encoding="utf-8"))
            matching = next(
                (row for row in profiles if int(row.get("base_model_id", -1)) == base_model_id),
                None,
            )
            if matching and matching.get("template"):
                metadata_template = Path(str(matching["template"]))
                candidates = [
                    metadata_template,
                    template_arg / metadata_template.name,
                    *candidates,
                ]
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "template_metadata_warning": repr(exc),
                        "base_model_profiles": str(profile_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    raise FileNotFoundError(
        f"No base template for base_model_id={base_model_id} in {template_arg}. "
        f"Tried: {', '.join(candidate.name for candidate in candidates)}."
    )


def summarize_batch(samples: list[dict]) -> dict:
    ok = [row for row in samples if row.get("normal_termination", False)]
    failed = [row for row in samples if not row.get("normal_termination", False)]
    by_template: dict[str, dict[str, int]] = {}
    for row in samples:
        template = str(row.get("template", "unknown"))
        slot = by_template.setdefault(template, {"ok": 0, "failed": 0})
        if row.get("normal_termination", False):
            slot["ok"] += 1
        else:
            slot["failed"] += 1
    return {
        "attempted": len(samples),
        "successful": len(ok),
        "failed": len(failed),
        "success_fraction": (len(ok) / len(samples)) if samples else 0.0,
        "by_template": by_template,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--febio")
    parser.add_argument("--keep-xplt", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    selected = samples[args.start : args.start + args.count]
    if args.dataset_id is None and selected:
        args.dataset_id = selected[0].dataset_id
    if args.dataset_id and args.runs == DEFAULT_RUNS:
        args.runs = args.runs / args.dataset_id
    febio = febio_executable(args.febio)

    batch_summary = {
        "dataset_id": args.dataset_id,
        "manifest": str(args.manifest),
        "start": args.start,
        "count": args.count,
        "samples": [],
    }

    workers = max(1, int(args.workers))
    print(
        json.dumps(
            {
                "run_batch_start": {
                    "count": len(selected),
                    "febio": febio,
                    "manifest": str(args.manifest),
                    "start": args.start,
                    "template": str(args.template),
                    "workers": workers,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if workers == 1:
        for sample in selected:
            sample_summary = run_one(sample, args, febio)
            batch_summary["samples"].append(sample_summary)
            print(json.dumps(sample_summary, sort_keys=True), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_sample = {pool.submit(run_one, sample, args, febio): sample for sample in selected}
            for future in as_completed(future_to_sample):
                sample_summary = future.result()
                batch_summary["samples"].append(sample_summary)
                print(json.dumps(sample_summary, sort_keys=True), flush=True)

        batch_summary["samples"].sort(key=lambda row: row["sample_id"])

    batch_name = f"batch_{args.start:06d}_{args.start + len(selected) - 1:06d}.json"
    batch_summary["health"] = summarize_batch(batch_summary["samples"])
    write_json(args.runs / batch_name, batch_summary)
    print(json.dumps({"batch_health": batch_summary["health"]}, sort_keys=True), flush=True)

    if args.pack:
        pack_dir = args.pack_dir or (ROOT / "shards" / args.dataset_id if args.dataset_id else args.runs.parent / "shards")
        pack_summary = pack_batch(
            runs=args.runs,
            start=args.start,
            count=len(selected),
            out_dir=pack_dir,
            dataset_id=args.dataset_id,
            cleanup=args.cleanup,
        )
        print(json.dumps({"packed": pack_summary["npz"], "cleanup": args.cleanup}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
