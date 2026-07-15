from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from optimize_packet_size import analyze_packet_sizes


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

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

REQUIRED_SHARD_KEYS = {
    "sample_ids",
    "sample_names",
    "params_json",
    "sole_near_element_mask",
    "last_nodes",
    "last_elements",
    "last_contact",
    "last_element_von_mises",
}

PINN_HISTORY_KEYS = {
    "node_times",
    "element_times",
    "contact_times",
    "node_history",
    "element_history",
    "contact_history",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_grid(face_count: int) -> np.ndarray:
    side = int(round(face_count ** 0.5))
    if side * side != face_count:
        x = np.linspace(-1.0, 1.0, face_count, dtype=np.float64)
        return np.stack([x, np.zeros_like(x)], axis=1)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, side, dtype=np.float64),
        np.linspace(-1.0, 1.0, side, dtype=np.float64),
        indexing="ij",
    )
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    vals = finite(values)
    if vals.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.quantile(vals, QUANTILES)
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p10": float(qs[3]),
        "p25": float(qs[4]),
        "p50": float(qs[5]),
        "p75": float(qs[6]),
        "p90": float(qs[7]),
        "p95": float(qs[8]),
        "p99": float(qs[9]),
        "max": float(qs[10]),
    }


def row_for(group: str, name: str, values: np.ndarray) -> dict[str, Any]:
    return {"group": group, "name": name, **distribution(values)}


def params_from_json(text: str) -> dict[str, float]:
    try:
        raw = json.loads(str(text))
    except json.JSONDecodeError:
        raw = {}
    return {name: float(raw.get(name, 0.0)) for name in PARAM_NAMES}


def collect_attempt_metadata(runs_dir: Path, shard_dir: Path) -> dict[str, Any]:
    attempted: set[int] = set()
    failed: dict[int, str] = {}
    normal_false = 0
    normal_true = 0

    for directory in [runs_dir, shard_dir]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("batch_*.json")):
            try:
                meta = load_json(path)
            except Exception:
                continue

            start = meta.get("start")
            count = meta.get("count")
            if isinstance(start, int) and isinstance(count, int):
                attempted.update(range(start, start + count))

            for item in meta.get("failed_samples", []) or []:
                sample_id = item.get("sample_id")
                if isinstance(sample_id, int):
                    failed[sample_id] = str(item.get("reason", "failed"))

            for item in (meta.get("results", []) or []) + (meta.get("samples", []) or []):
                sample_id = item.get("sample_id")
                if isinstance(sample_id, int):
                    attempted.add(sample_id)
                    if item.get("normal_termination") is False:
                        normal_false += 1
                        failed.setdefault(sample_id, "normal_termination_false")
                    elif item.get("normal_termination") is True:
                        normal_true += 1

            for item in meta.get("summaries", []) or []:
                sample_name = str(item.get("sample_name", ""))
                sample_id = None
                if sample_name.startswith("sample_"):
                    try:
                        sample_id = int(sample_name.split("_")[-1])
                    except ValueError:
                        sample_id = None
                if sample_id is not None:
                    attempted.add(sample_id)
                    if item.get("normal_termination") is False:
                        normal_false += 1
                        failed.setdefault(sample_id, "normal_termination_false")
                    elif item.get("normal_termination") is True:
                        normal_true += 1

    return {
        "attempted_sample_ids": sorted(attempted),
        "failed_sample_ids": sorted(failed),
        "failed_reasons": failed,
        "json_normal_true_rows": normal_true,
        "json_normal_false_rows": normal_false,
    }


def summarize_shards(
    shard_dir: Path,
    require_history: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, list[float]]]]:
    shards = sorted(shard_dir.glob("batch_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No batch_*.npz files found in {shard_dir}")

    overall_rows: list[dict[str, Any]] = []
    base_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall_values: dict[str, list[float]] = defaultdict(list)
    valid_ids: list[int] = []
    invalid_inside_shard: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    missing_required_by_shard: dict[str, list[str]] = {}
    missing_history_by_shard: dict[str, list[str]] = {}
    dataset_ids_seen: set[str] = set()

    regression_params: list[np.ndarray] = []
    regression_pressure: list[np.ndarray] = []
    regression_base_ids: list[int] = []
    regression_sample_ids: list[int] = []

    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            files = set(data.files)
            missing_required = sorted(REQUIRED_SHARD_KEYS - files)
            missing_history = sorted(PINN_HISTORY_KEYS - files)
            if missing_required:
                missing_required_by_shard[str(shard)] = missing_required
            if missing_history:
                missing_history_by_shard[str(shard)] = missing_history
            if "dataset_ids" in files:
                dataset_ids_seen.update(str(value) for value in data["dataset_ids"].astype(str).tolist())
            schema_rows.append(
                {
                    "shard": str(shard),
                    "has_dataset_ids": "dataset_ids" in files,
                    "has_all_required_fields": not missing_required,
                    "has_all_pinn_history_fields": not missing_history,
                    "node_history_shape": list(data["node_history"].shape) if "node_history" in files else None,
                    "element_history_shape": list(data["element_history"].shape) if "element_history" in files else None,
                    "contact_history_shape": list(data["contact_history"].shape) if "contact_history" in files else None,
                    "node_history_valid_mask": "node_history_valid_mask" in files or "node_times_valid_mask" in files,
                    "element_history_valid_mask": "element_history_valid_mask" in files or "element_times_valid_mask" in files,
                    "contact_history_valid_mask": "contact_history_valid_mask" in files or "contact_times_valid_mask" in files,
                }
            )
            if missing_required:
                continue
            sample_ids = data["sample_ids"].astype(int)
            params_json = data["params_json"].astype(str) if "params_json" in data.files else np.asarray(["{}"] * len(sample_ids))
            contact = np.asarray(data["last_contact"], dtype=np.float64)
            pressure = contact[:, :, 2]
            gap = contact[:, :, 1]
            nodes = np.asarray(data["last_nodes"], dtype=np.float64)
            elements = np.asarray(data["last_elements"], dtype=np.float64)
            von_mises = np.asarray(data["last_element_von_mises"], dtype=np.float64)
            sole_mask = np.asarray(data["sole_near_element_mask"], dtype=bool) if "sole_near_element_mask" in data.files else None

            grid = contact_grid(pressure.shape[1])
            reaction_proxy = np.sum(pressure, axis=1)
            peak_pressure = np.max(pressure, axis=1)
            mean_pressure = np.mean(pressure, axis=1)
            active_contact_faces = np.sum(pressure > 1e-10, axis=1)
            weights = np.maximum(pressure, 0.0)
            denom = np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
            center = weights @ grid / denom
            displacement_mag = np.linalg.norm(nodes[:, :, 4:7], axis=2)
            max_displacement = np.max(displacement_mag, axis=1)
            mean_displacement = np.mean(displacement_mag, axis=1)
            max_von_mises = np.max(von_mises, axis=1)
            mean_von_mises = np.mean(von_mises, axis=1)
            stress_mag = np.linalg.norm(elements[:, :, 4:10], axis=2)
            max_stress_mag = np.max(stress_mag, axis=1)
            sole_count = np.sum(sole_mask, axis=1) if sole_mask is not None else np.full(len(sample_ids), np.nan)

            finite_mask = (
                np.all(np.isfinite(pressure), axis=1)
                & np.all(np.isfinite(nodes.reshape(len(sample_ids), -1)), axis=1)
                & np.all(np.isfinite(elements.reshape(len(sample_ids), -1)), axis=1)
                & np.all(np.isfinite(von_mises), axis=1)
            )

            shard_valid = int(np.sum(finite_mask))
            shard_rows.append(
                {
                    "shard": str(shard),
                    "packed_samples": int(len(sample_ids)),
                    "finite_samples": shard_valid,
                    "invalid_finite_samples": int(len(sample_ids) - shard_valid),
                }
            )

            for local_i, sample_id in enumerate(sample_ids.tolist()):
                params = params_from_json(params_json[local_i])
                base_id = int(round(params.get("base_model_id", 0.0)))
                if not bool(finite_mask[local_i]):
                    invalid_inside_shard.append({"sample_id": int(sample_id), "shard": str(shard), "reason": "nonfinite_values"})
                    continue

                valid_ids.append(int(sample_id))
                metrics = {
                    "reaction_proxy": reaction_proxy[local_i],
                    "peak_pressure": peak_pressure[local_i],
                    "mean_pressure": mean_pressure[local_i],
                    "active_contact_faces": active_contact_faces[local_i],
                    "center_of_pressure_x": center[local_i, 0],
                    "center_of_pressure_y": center[local_i, 1],
                    "min_gap": np.min(gap[local_i]),
                    "mean_gap": np.mean(gap[local_i]),
                    "max_displacement": max_displacement[local_i],
                    "mean_displacement": mean_displacement[local_i],
                    "max_von_mises": max_von_mises[local_i],
                    "mean_von_mises": mean_von_mises[local_i],
                    "max_stress_magnitude": max_stress_mag[local_i],
                    "sole_near_element_count": sole_count[local_i],
                }
                for name, value in metrics.items():
                    if math.isfinite(float(value)):
                        overall_values[name].append(float(value))
                        base_values[base_id][name].append(float(value))
                for name, value in params.items():
                    overall_values[f"param/{name}"].append(float(value))
                    base_values[base_id][f"param/{name}"].append(float(value))

                regression_params.append(np.asarray([params[name] for name in PARAM_NAMES], dtype=np.float64))
                regression_pressure.append(pressure[local_i].astype(np.float64))
                regression_base_ids.append(base_id)
                regression_sample_ids.append(int(sample_id))

    for name, values in sorted(overall_values.items()):
        overall_rows.append(row_for("all", name, np.asarray(values, dtype=np.float64)))
    by_base_rows = []
    for base_id in sorted(base_values):
        for name, values in sorted(base_values[base_id].items()):
            by_base_rows.append(row_for(f"base_{base_id:02d}", name, np.asarray(values, dtype=np.float64)))

    arrays = {
        "sample_ids": np.asarray(regression_sample_ids, dtype=np.int64),
        "base_ids": np.asarray(regression_base_ids, dtype=np.int64),
        "params": np.stack(regression_params, axis=0) if regression_params else np.zeros((0, len(PARAM_NAMES))),
        "pressure": np.stack(regression_pressure, axis=0) if regression_pressure else np.zeros((0, 256)),
    }
    summary = {
        "shards_found": len(shards),
        "dataset_ids_seen": sorted(dataset_ids_seen),
        "shards_with_all_required_fields": int(sum(row["has_all_required_fields"] for row in schema_rows)),
        "shards_with_all_pinn_history_fields": int(sum(row["has_all_pinn_history_fields"] for row in schema_rows)),
        "missing_required_fields_by_shard": missing_required_by_shard,
        "missing_pinn_history_fields_by_shard": missing_history_by_shard,
        "schema": schema_rows,
        "require_history": require_history,
        "valid_samples_in_shards": len(valid_ids),
        "valid_sample_ids_min": int(min(valid_ids)) if valid_ids else None,
        "valid_sample_ids_max": int(max(valid_ids)) if valid_ids else None,
        "invalid_samples_inside_shards": invalid_inside_shard,
        "shards": shard_rows,
        "regression_arrays": arrays,
    }
    if missing_required_by_shard:
        raise ValueError(f"Some shards are missing required fields: {missing_required_by_shard}")
    if require_history and missing_history_by_shard:
        raise ValueError(f"Some shards are missing full-PINN history fields: {missing_history_by_shard}")
    return summary, overall_rows, by_base_rows, base_values


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def per_sample_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_mean = np.mean(y_true, axis=1, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=1)
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    mask = ss_tot > 1e-12
    out[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, base_ids: np.ndarray) -> dict[str, Any]:
    sample_r2 = per_sample_r2(y_true, y_pred)
    reaction_true = np.sum(y_true, axis=1)
    reaction_pred = np.sum(y_pred, axis=1)
    peak_true = np.max(y_true, axis=1)
    peak_pred = np.max(y_pred, axis=1)
    grid = contact_grid(y_true.shape[1])
    wt = np.maximum(y_true, 0.0)
    wp = np.maximum(y_pred, 0.0)
    center_true = wt @ grid / np.maximum(np.sum(wt, axis=1, keepdims=True), 1e-12)
    center_pred = wp @ grid / np.maximum(np.sum(wp, axis=1, keepdims=True), 1e-12)

    out: dict[str, Any] = {
        "pressure_r2": r2_score(y_true.reshape(-1), y_pred.reshape(-1)),
        "pressure_rmse": rmse(y_true, y_pred),
        "pressure_mae": mae(y_true, y_pred),
        "pressure_sample_r2_p10": float(np.nanquantile(sample_r2, 0.10)),
        "pressure_sample_r2_p50": float(np.nanquantile(sample_r2, 0.50)),
        "pressure_sample_r2_p90": float(np.nanquantile(sample_r2, 0.90)),
        "reaction_proxy_r2": r2_score(reaction_true, reaction_pred),
        "reaction_proxy_rmse": rmse(reaction_true, reaction_pred),
        "reaction_proxy_mae": mae(reaction_true, reaction_pred),
        "peak_pressure_r2": r2_score(peak_true, peak_pred),
        "peak_pressure_rmse": rmse(peak_true, peak_pred),
        "peak_pressure_mae": mae(peak_true, peak_pred),
        "center_of_pressure_rmse": rmse(center_true, center_pred),
        "center_of_pressure_mae": mae(center_true, center_pred),
        "n": int(y_true.shape[0]),
    }
    basewise = {}
    for base_id in sorted(set(base_ids.tolist())):
        mask = base_ids == base_id
        if np.sum(mask) < 2:
            continue
        basewise[f"base_{base_id:02d}"] = {
            "pressure_r2": r2_score(y_true[mask].reshape(-1), y_pred[mask].reshape(-1)),
            "pressure_rmse": rmse(y_true[mask], y_pred[mask]),
            "n": int(np.sum(mask)),
        }
    out["by_base"] = basewise
    return out


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    xs = (x_train - x_mean) / x_std
    xt = (x_test - x_mean) / x_std
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    yc = y_train - y_mean
    reg = alpha * np.eye(xs.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xs.T @ xs + reg, xs.T @ yc)
    return xt @ coef + y_mean


def run_regression_baselines(
    arrays: dict[str, np.ndarray],
    max_samples: int,
    seed: int,
    alphas: list[float],
) -> dict[str, Any]:
    sample_ids = arrays["sample_ids"]
    base_ids = arrays["base_ids"]
    x_all = arrays["params"]
    y_all = arrays["pressure"]
    if y_all.shape[0] < 20:
        return {"skipped": True, "reason": "fewer_than_20_valid_samples"}

    rng = np.random.default_rng(seed)
    chosen = np.arange(y_all.shape[0])
    if max_samples and len(chosen) > max_samples:
        chosen = rng.choice(chosen, size=max_samples, replace=False)
        chosen.sort()
    sample_ids = sample_ids[chosen]
    base_ids = base_ids[chosen]
    x_all = x_all[chosen]
    y_all = y_all[chosen]

    seen_mask = base_ids < 10
    unseen_mask = base_ids >= 10
    seen_indices = np.where(seen_mask)[0]
    rng.shuffle(seen_indices)
    val_count = max(1, int(round(len(seen_indices) * 0.2)))
    val_seen = seen_indices[:val_count]
    train_seen = seen_indices[val_count:]
    unseen = np.where(unseen_mask)[0]

    feature_sets = {
        "params_all": np.arange(x_all.shape[1]),
        "params_without_base_id": np.asarray([i for i, name in enumerate(PARAM_NAMES) if name != "base_model_id"], dtype=int),
        "physics_only_no_base_profile": np.asarray(
            [i for i, name in enumerate(PARAM_NAMES) if not name.startswith("base_")], dtype=int
        ),
    }

    out: dict[str, Any] = {
        "valid_samples_used": int(y_all.shape[0]),
        "train_seen_base_samples": int(len(train_seen)),
        "random_seen_validation_samples": int(len(val_seen)),
        "unseen_base_validation_samples": int(len(unseen)),
        "sample_id_min": int(np.min(sample_ids)),
        "sample_id_max": int(np.max(sample_ids)),
        "baselines": {},
    }

    splits = {"random_seen_base_split": val_seen}
    if len(unseen) > 0:
        splits["unseen_base_10_11_split"] = unseen

    for split_name, test_idx in splits.items():
        y_train = y_all[train_seen]
        y_test = y_all[test_idx]
        mean_pred = np.repeat(np.mean(y_train, axis=0, keepdims=True), len(test_idx), axis=0)
        out["baselines"][split_name] = {
            "mean_pressure_map": regression_metrics(y_test, mean_pred, base_ids[test_idx])
        }

        for feature_name, feature_idx in feature_sets.items():
            x_train = x_all[train_seen][:, feature_idx]
            x_test = x_all[test_idx][:, feature_idx]
            candidates = []
            for alpha in alphas:
                pred = ridge_predict(x_train, y_train, x_test, alpha=alpha)
                metrics = regression_metrics(y_test, pred, base_ids[test_idx])
                metrics["alpha"] = alpha
                candidates.append(metrics)
            best = max(candidates, key=lambda item: item["pressure_r2"] if math.isfinite(item["pressure_r2"]) else -1e30)
            out["baselines"][split_name][f"ridge_{feature_name}"] = best

    return out


def valid_counts_by_base(
    arrays: dict[str, np.ndarray],
    attempted_ids: list[int],
    failed_ids: list[int],
    failed_not_packed_ids: list[int],
) -> list[dict[str, Any]]:
    rows = []
    base_ids = arrays["base_ids"]
    for base_id in sorted(set(base_ids.tolist())):
        rows.append(
            {
                "base_model_id": int(base_id),
                "valid_samples": int(np.sum(base_ids == base_id)),
            }
        )
    rows.append(
        {
            "base_model_id": "all",
            "valid_samples": int(len(base_ids)),
            "attempted_samples_from_json": int(len(attempted_ids)),
            "failed_samples_from_json_raw": int(len(failed_ids)),
            "failed_samples_not_packed": int(len(failed_not_packed_ids)),
            "apparent_success_rate": float(len(base_ids) / (len(base_ids) + len(failed_not_packed_ids)))
            if (len(base_ids) + len(failed_not_packed_ids))
            else None,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset quality checks for packed FEBio foot shards.")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--shard-dir", type=Path, default=Path("shards"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("training/dataset_quality"))
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--max-regression-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--packet-size-analysis", action="store_true")
    parser.add_argument(
        "--packet-runtime-input",
        type=Path,
        action="append",
        default=[],
        help="Optional batch JSON, SLURM log, or directory with per-sample durations. Defaults to shard/runs dirs.",
    )
    parser.add_argument("--packet-workers", type=int, default=64)
    parser.add_argument("--packet-array-concurrency", type=int, default=4)
    parser.add_argument("--packet-min", type=int, default=64)
    parser.add_argument("--packet-max", type=int, default=256)
    parser.add_argument("--packet-target-attempts", type=int, default=None)
    parser.add_argument("--packet-target-successes", type=int, default=None)
    parser.add_argument("--packet-bootstrap", type=int, default=500)
    parser.add_argument("--packet-overhead-sec", type=float, default=0.0)
    args = parser.parse_args()
    if args.dataset_id:
        if args.shard_dir == Path("shards"):
            args.shard_dir = Path("shards") / args.dataset_id
        if args.runs_dir == Path("runs"):
            args.runs_dir = Path("runs") / args.dataset_id
        if args.out_dir == Path("training/dataset_quality"):
            args.out_dir = Path("training/dataset_quality") / args.dataset_id

    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempt_meta = collect_attempt_metadata(args.runs_dir, args.shard_dir)
    shard_summary, overall_dist_rows, by_base_dist_rows, _ = summarize_shards(args.shard_dir, require_history=args.require_history)
    arrays = shard_summary.pop("regression_arrays")
    regression = run_regression_baselines(
        arrays=arrays,
        max_samples=args.max_regression_samples,
        seed=args.seed,
        alphas=args.ridge_alpha,
    )
    packet_size_result = None
    if args.packet_size_analysis:
        packet_inputs = args.packet_runtime_input or [args.shard_dir, args.runs_dir]
        packet_target_attempts = args.packet_target_attempts
        if packet_target_attempts is None and args.packet_target_successes is None:
            packet_target_attempts = len(attempt_meta["attempted_sample_ids"]) or int(arrays["pressure"].shape[0])
        packet_size_result = analyze_packet_sizes(
            inputs=packet_inputs,
            workers=args.packet_workers,
            array_concurrency=args.packet_array_concurrency,
            min_packet=args.packet_min,
            max_packet=args.packet_max,
            target_attempts=packet_target_attempts,
            target_successes=args.packet_target_successes,
            bootstrap=args.packet_bootstrap,
            overhead_sec=args.packet_overhead_sec,
            seed=args.seed,
            out_dir=args.out_dir / "packet_size",
        )

    attempted_ids = attempt_meta["attempted_sample_ids"]
    failed_ids = attempt_meta["failed_sample_ids"]
    valid_id_set = set(int(x) for x in arrays["sample_ids"].tolist())
    failed_id_set = set(int(x) for x in failed_ids)
    failed_not_packed_ids = sorted(failed_id_set - valid_id_set)
    failed_but_packed_ids = sorted(failed_id_set & valid_id_set)
    counts_rows = valid_counts_by_base(arrays, attempted_ids, failed_ids, failed_not_packed_ids)
    summary = {
        "shard_dir": str(args.shard_dir),
        "runs_dir": str(args.runs_dir),
        "out_dir": str(args.out_dir),
        "dataset_id": args.dataset_id,
        **shard_summary,
        "attempted_samples_from_json": len(attempted_ids),
        "failed_samples_from_json_raw": len(failed_ids),
        "failed_samples_not_packed": len(failed_not_packed_ids),
        "failed_samples_also_present_in_packed_shards": len(failed_but_packed_ids),
        "apparent_success_rate": float(arrays["pressure"].shape[0] / (arrays["pressure"].shape[0] + len(failed_not_packed_ids)))
        if (arrays["pressure"].shape[0] + len(failed_not_packed_ids))
        else None,
        "json_normal_true_rows": attempt_meta["json_normal_true_rows"],
        "json_normal_false_rows": attempt_meta["json_normal_false_rows"],
        "regression": regression,
        "packet_size_optimization": packet_size_result["summary"] if packet_size_result else None,
    }

    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "regression_baselines.json", regression)
    write_csv(args.out_dir / "distributions_overall.csv", overall_dist_rows)
    write_csv(args.out_dir / "distributions_by_base.csv", by_base_dist_rows)
    write_csv(args.out_dir / "valid_counts_by_base.csv", counts_rows)
    write_csv(args.out_dir / "shard_schema.csv", shard_summary["schema"])
    write_json(
        args.out_dir / "failed_samples_from_json.json",
        {
            "failed_sample_ids_raw": failed_ids,
            "failed_sample_ids_not_packed": failed_not_packed_ids,
            "failed_sample_ids_also_present_in_packed_shards": failed_but_packed_ids,
            "failed_reasons": attempt_meta["failed_reasons"],
        },
    )

    print(json.dumps({k: v for k, v in summary.items() if k not in {"shards", "invalid_samples_inside_shards", "regression"}}, indent=2))
    print(f"[OK] wrote dataset quality report to {args.out_dir}")


if __name__ == "__main__":
    main()
