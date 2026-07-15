from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


FIXED_ARRAYS = [
    "last_nodes",
    "last_elements",
    "last_contact",
    "last_element_von_mises",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_batch(
    runs: Path,
    start: int,
    count: int,
    out_dir: Path,
    cleanup: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    end = start + count - 1
    out_npz = out_dir / f"batch_{start:06d}_{end:06d}.npz"
    out_json = out_dir / f"batch_{start:06d}_{end:06d}.json"

    sample_ids: list[int] = []
    sample_names: list[str] = []
    params: list[dict] = []
    summaries: list[dict] = []
    failures: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in FIXED_ARRAYS}
    sole_masks: list[np.ndarray] = []

    for sample_id in range(start, start + count):
        sample_name = f"sample_{sample_id:06d}"
        run_dir = runs / sample_name
        summary_path = run_dir / "summary.json"
        params_path = run_dir / "params.json"
        dataset_path = run_dir / f"{sample_name}_dataset.npz"

        if not summary_path.exists():
            failures.append({"sample_id": sample_id, "reason": "missing_summary"})
            continue

        summary = load_json(summary_path)
        if not summary.get("normal_termination", False) or "extract_error" in summary:
            failures.append({"sample_id": sample_id, "reason": "failed_or_unextracted", "summary": summary})
            continue

        if not dataset_path.exists():
            failures.append({"sample_id": sample_id, "reason": "missing_dataset"})
            continue

        data = np.load(dataset_path)
        for key in FIXED_ARRAYS:
            arrays[key].append(data[key])

        n_elements = data["last_elements"].shape[0]
        mask = np.zeros(n_elements, dtype=bool)
        mask[data["sole_near_element_indices"]] = True
        sole_masks.append(mask)

        sample_ids.append(sample_id)
        sample_names.append(sample_name)
        params.append(load_json(params_path)["params"] if params_path.exists() else {})
        summaries.append(summary)

    if not sample_ids:
        raise RuntimeError(f"No successful samples found in range {start}:{start + count}")

    packed = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "sample_names": np.asarray(sample_names),
        "params_json": np.asarray([json.dumps(p, sort_keys=True) for p in params]),
        "sole_near_element_mask": np.stack(sole_masks, axis=0),
    }
    for key, values in arrays.items():
        packed[key] = np.stack(values, axis=0)

    np.savez_compressed(out_npz, **packed)

    meta = {
        "start": start,
        "count": count,
        "end": end,
        "packed_samples": len(sample_ids),
        "failed_samples": failures,
        "npz": str(out_npz),
        "summaries": summaries,
        "cleanup": cleanup,
    }
    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    if cleanup:
        successful = set(sample_names)
        for sample_name in successful:
            shutil.rmtree(runs / sample_name)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    meta = pack_batch(
        runs=args.runs,
        start=args.start,
        count=args.count,
        out_dir=args.out_dir,
        cleanup=args.cleanup,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "summaries"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

