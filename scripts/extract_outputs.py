from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from common import von_mises


TIME_RE = re.compile(r"\*Time\s*=\s*([-+0-9.eE]+)")


def read_numeric_records(path: Path) -> list[tuple[float | None, np.ndarray]]:
    records: list[tuple[float | None, np.ndarray]] = []
    time_value = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows, time_value
        if rows:
            records.append((time_value, np.asarray(rows, dtype=np.float32)))
            rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            match = TIME_RE.match(text)
            if match:
                flush()
                time_value = float(match.group(1))
                continue
            if text.startswith("*"):
                continue
            parts = [p for p in text.replace(" ", ",").split(",") if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    flush()
    return records


def collect_series(run_dir: Path, prefix: str) -> tuple[np.ndarray, list[np.ndarray]]:
    paths = sorted(Path(p) for p in glob.glob(str(run_dir / f"{prefix}*")))
    times = []
    arrays = []
    for path in paths:
        for time_value, arr in read_numeric_records(path):
            if arr.size == 0:
                continue
            times.append(-1.0 if time_value is None else time_value)
            arrays.append(arr)
    return np.asarray(times, dtype=np.float32), arrays


def stack_history(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0, 0), dtype=np.float32)
    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        raise ValueError("Cannot pack history with changing output shapes")
    return np.stack(arrays, axis=0).astype(np.float32)


def extract(run_dir: Path, sample_name: str, include_history: bool = False) -> dict[str, object]:
    out_npz = run_dir / f"{sample_name}_dataset.npz"

    node_t, node_arrays = collect_series(run_dir, f"{sample_name}_nodes.csv")
    elem_t, elem_arrays = collect_series(run_dir, f"{sample_name}_elements.csv")
    contact_t, contact_arrays = collect_series(run_dir, f"{sample_name}_contact.csv")

    if not elem_arrays:
        raise FileNotFoundError(f"No element output files found in {run_dir}")

    last_elements = elem_arrays[-1]
    # columns: id, x, y, z, sx, sy, sz, sxy, syz, sxz
    stresses = last_elements[:, 4:10]
    vm = np.asarray([von_mises(*row) for row in stresses], dtype=np.float32)
    sole_near = np.where(last_elements[:, 3] <= 0.22)[0].astype(np.int32)

    payload = {
        "node_times": node_t,
        "element_times": elem_t,
        "contact_times": contact_t,
        "last_nodes": node_arrays[-1] if node_arrays else np.zeros((0, 0), dtype=np.float32),
        "last_elements": last_elements,
        "last_contact": contact_arrays[-1] if contact_arrays else np.zeros((0, 0), dtype=np.float32),
        "last_element_von_mises": vm,
        "sole_near_element_indices": sole_near,
    }
    if include_history:
        payload.update(
            {
                "node_history": stack_history(node_arrays),
                "element_history": stack_history(elem_arrays),
                "contact_history": stack_history(contact_arrays),
            }
        )
    np.savez_compressed(out_npz, **payload)
    return {
        "dataset": str(out_npz),
        "node_steps": len(node_arrays),
        "element_steps": len(elem_arrays),
        "contact_steps": len(contact_arrays),
        "element_count": int(last_elements.shape[0]),
        "sole_near_element_count": int(sole_near.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    summary = extract(args.run_dir, args.sample_name, include_history=args.include_history)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary:
        args.summary.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
