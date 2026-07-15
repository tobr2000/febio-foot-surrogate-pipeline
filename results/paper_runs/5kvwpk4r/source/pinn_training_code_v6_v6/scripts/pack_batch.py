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

OPTIONAL_ARRAYS = [
    "node_times",
    "element_times",
    "contact_times",
    "node_history",
    "element_history",
    "contact_history",
]

OPTIONAL_TIME_KEYS = {"node_times", "element_times", "contact_times"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pad_optional_stack(values: list[np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length time histories with NaN padding and a valid-time mask."""
    if not values:
        raise ValueError(f"No values for optional key {key}")
    max_steps = max(int(value.shape[0]) for value in values)
    if key in OPTIONAL_TIME_KEYS:
        out = np.full((len(values), max_steps), np.nan, dtype=np.float32)
        mask = np.zeros((len(values), max_steps), dtype=bool)
        for i, value in enumerate(values):
            steps = int(value.shape[0])
            out[i, :steps] = value.astype(np.float32)
            mask[i, :steps] = True
        return out, mask

    tail_shape = values[0].shape[1:]
    if any(value.shape[1:] != tail_shape for value in values):
        raise ValueError(f"Optional array {key} has incompatible non-time shapes")
    out = np.full((len(values), max_steps, *tail_shape), np.nan, dtype=np.float32)
    mask = np.zeros((len(values), max_steps), dtype=bool)
    for i, value in enumerate(values):
        steps = int(value.shape[0])
        out[i, :steps] = value.astype(np.float32)
        mask[i, :steps] = True
    return out, mask


def pack_batch(
    runs: Path,
    start: int,
    count: int,
    out_dir: Path,
    dataset_id: str | None = None,
    cleanup: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    end = start + count - 1
    out_npz = out_dir / f"batch_{start:06d}_{end:06d}.npz"
    out_json = out_dir / f"batch_{start:06d}_{end:06d}.json"

    sample_ids: list[int] = []
    sample_names: list[str] = []
    dataset_ids: list[str] = []
    params: list[dict] = []
    summaries: list[dict] = []
    failures: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in FIXED_ARRAYS}
    optional_arrays: dict[str, list[np.ndarray]] = {key: [] for key in OPTIONAL_ARRAYS}
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
        for key in OPTIONAL_ARRAYS:
            if key in data.files:
                optional_arrays[key].append(data[key])

        n_elements = data["last_elements"].shape[0]
        mask = np.zeros(n_elements, dtype=bool)
        mask[data["sole_near_element_indices"]] = True
        sole_masks.append(mask)

        params_payload = load_json(params_path) if params_path.exists() else {}
        row_dataset_id = str(params_payload.get("dataset_id") or dataset_id or "default")
        sample_ids.append(sample_id)
        sample_names.append(sample_name)
        dataset_ids.append(row_dataset_id)
        params.append(params_payload.get("params", {}))
        summaries.append(summary)

    if not sample_ids:
        raise RuntimeError(f"No successful samples found in range {start}:{start + count}")

    packed = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "sample_names": np.asarray(sample_names),
        "dataset_ids": np.asarray(dataset_ids),
        "params_json": np.asarray([json.dumps(p, sort_keys=True) for p in params]),
        "sole_near_element_mask": np.stack(sole_masks, axis=0),
    }
    for key, values in arrays.items():
        packed[key] = np.stack(values, axis=0)
    for key, values in optional_arrays.items():
        if len(values) == len(sample_ids):
            try:
                stacked, valid_mask = pad_optional_stack(values, key)
                packed[key] = stacked
                packed[f"{key}_valid_mask"] = valid_mask
            except ValueError:
                failures.append({"reason": f"ragged_optional_array_{key}", "samples": len(values)})

    np.savez_compressed(out_npz, **packed)

    meta = {
        "dataset_id": dataset_id or (dataset_ids[0] if dataset_ids else "default"),
        "schema_version": 2,
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
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    meta = pack_batch(
        runs=args.runs,
        start=args.start,
        count=args.count,
        out_dir=args.out_dir,
        dataset_id=args.dataset_id,
        cleanup=args.cleanup,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "summaries"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
