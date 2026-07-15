from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (
    DEFAULT_RUNS,
    DEFAULT_TEMPLATE,
    febio_executable,
    load_manifest,
    normal_termination,
    render_feb,
    run_febio,
    write_json,
)
from extract_outputs import extract
from pack_batch import pack_batch


def run_one(sample, args, febio: str) -> dict:
    run_dir = args.runs / sample.name
    run_dir.mkdir(parents=True, exist_ok=True)
    feb_path = run_dir / f"{sample.name}.feb"
    template, include_base_profile = resolve_template(args.template, sample)
    render_feb(template, sample, feb_path, include_base_profile=include_base_profile)
    write_json(run_dir / "params.json", {"sample_id": sample.sample_id, "params": sample.params})

    result = run_febio(febio, feb_path, run_dir)
    solver_log = run_dir / f"{sample.name}_solver_stdout.log"
    solver_log.write_text(result.stdout, encoding="utf-8", errors="ignore")
    ok = result.returncode == 0 and normal_termination(result.stdout)

    sample_summary = {
        "sample_id": sample.sample_id,
        "name": sample.name,
        "returncode": result.returncode,
        "normal_termination": ok,
        "run_dir": str(run_dir),
        "template": str(template),
    }

    if ok:
        try:
            sample_summary.update(extract(run_dir, sample.name))
        except Exception as exc:
            sample_summary["extract_error"] = repr(exc)
    else:
        sample_summary["solver_log"] = str(solver_log)

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
    candidates = [
        template_arg / f"simplefoot_base_{base_model_id:02d}.feb",
        template_arg / f"base_{base_model_id:02d}.feb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    raise FileNotFoundError(
        f"No base template for base_model_id={base_model_id} in {template_arg}. "
        f"Expected {candidates[0].name}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--febio")
    parser.add_argument("--keep-xplt", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    selected = samples[args.start : args.start + args.count]
    febio = febio_executable(args.febio)

    batch_summary = {
        "manifest": str(args.manifest),
        "start": args.start,
        "count": args.count,
        "samples": [],
    }

    workers = max(1, int(args.workers))
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
    write_json(args.runs / batch_name, batch_summary)

    if args.pack:
        pack_dir = args.pack_dir or (args.runs.parent / "shards")
        pack_summary = pack_batch(
            runs=args.runs,
            start=args.start,
            count=len(selected),
            out_dir=pack_dir,
            cleanup=args.cleanup,
        )
        print(json.dumps({"packed": pack_summary["npz"], "cleanup": args.cleanup}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
