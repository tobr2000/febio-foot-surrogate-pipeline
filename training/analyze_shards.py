from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PARAM_NAMES = [
    "base_model_id",
    "base_is_training_family",
    "base_foot_length",
    "base_foot_width",
    "base_arch_lift",
    "base_leg_length",
    "base_toe_splay",
    "scale_x",
    "scale_y",
    "scale_z",
    "E_flesh",
    "E_bone",
    "E_joint",
    "E_collar",
    "E_heel",
    "E_forefoot",
    "E_plantar",
    "E_achilles",
    "friction",
    "early_down_disp",
    "peak_down_disp",
    "final_down_disp",
    "forward_disp",
    "peak_time",
    "lateral_disp",
    "toe_off_bias",
    "heel_toe_roll",
    "arch_lift",
]


def _finite_report(name: str, array: np.ndarray) -> dict:
    numeric = np.asarray(array)
    finite = np.isfinite(numeric)
    return {
        "name": name,
        "shape": list(numeric.shape),
        "dtype": str(numeric.dtype),
        "finite_fraction": float(finite.mean()) if finite.size else 1.0,
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
        "std": float(np.nanstd(values)),
    }


def analyze_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    report: dict = {"path": str(path), "arrays": []}
    for key in data.files:
        if np.issubdtype(data[key].dtype, np.number) or data[key].dtype == np.bool_:
            report["arrays"].append(_finite_report(key, data[key]))
        else:
            report["arrays"].append(
                {"name": key, "shape": list(data[key].shape), "dtype": str(data[key].dtype)}
            )

    nodes = data["last_nodes"]
    elements = data["last_elements"]
    contact = data["last_contact"]
    vm = data["last_element_von_mises"]
    sole_mask = data["sole_near_element_mask"]

    disp = nodes[:, :, 4:7]
    disp_mag = np.linalg.norm(disp, axis=-1)
    stress = elements[:, :, 4:10]
    pressure = contact[:, :, 2]
    gap = contact[:, :, 1]

    report["sample_count"] = int(nodes.shape[0])
    report["nodes_per_sample"] = int(nodes.shape[1])
    report["elements_per_sample"] = int(elements.shape[1])
    report["contact_faces_per_sample"] = int(contact.shape[1])
    report["displacement_magnitude"] = _summary(disp_mag)
    report["per_sample_max_displacement"] = _summary(np.max(disp_mag, axis=1))
    report["von_mises"] = _summary(vm)
    report["per_sample_max_von_mises"] = _summary(np.max(vm, axis=1))
    report["contact_gap"] = _summary(gap)
    report["contact_pressure"] = _summary(pressure)
    report["per_sample_max_contact_pressure"] = _summary(np.max(pressure, axis=1))
    report["sole_near_element_count"] = _summary(np.sum(sole_mask, axis=1))
    report["stress_component_mean"] = np.mean(stress, axis=(0, 1)).astype(float).tolist()
    report["stress_component_std"] = np.std(stress, axis=(0, 1)).astype(float).tolist()

    params = [json.loads(x) for x in data["params_json"].tolist()]
    param_report = {}
    for name in PARAM_NAMES:
        values = np.asarray([p.get(name, np.nan) for p in params], dtype=np.float64)
        param_report[name] = _summary(values)
    report["parameters"] = param_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize packed FEBio ML shard data.")
    parser.add_argument("--shard-dir", default="shards_test", help="Directory with batch_*.npz files.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    paths = sorted(shard_dir.glob("batch_*.npz"))
    if not paths:
        raise SystemExit(f"No batch_*.npz files found in {shard_dir}")

    reports = [analyze_npz(path) for path in paths]
    output = {"shard_count": len(reports), "shards": reports}
    text = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
