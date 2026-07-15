from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def better(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    if mode == "max":
        return candidate > incumbent
    raise ValueError(f"Unknown mode: {mode}")


def metric_mode(metric: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    lowered = metric.lower()
    if lowered.endswith("/r2") or lowered.endswith("_r2") or "pearson" in lowered or "spearman" in lowered:
        return "max"
    return "min"


def reselect_foot(
    run_metrics_path: Path,
    selection_metric: str,
    out_dir: Path,
) -> None:
    rows = read_csv(run_metrics_path)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset_label"], row["method"], row["run_id"])
        grouped.setdefault(key, []).append(row)

    mode = metric_mode(selection_metric)
    checkpoints = []
    metric_rows = []

    for (dataset_label, method, run_id), group in sorted(grouped.items()):
        selected_value: float | None = None
        selected_step: str | None = None
        selected_epoch: str | None = None
        selected_runtime: str | None = None
        for row in group:
            if row.get("metric") != selection_metric:
                continue
            value = as_float(row.get("value"))
            if value is None:
                continue
            if better(value, selected_value, mode):
                selected_value = value
                selected_step = row.get("step")
                selected_epoch = row.get("epoch")
                selected_runtime = row.get("runtime_seconds")
        if selected_value is None or selected_step is None:
            continue

        checkpoints.append(
            {
                "run_id": run_id,
                "dataset_label": dataset_label,
                "method": method,
                "selection_metric": selection_metric,
                "selection_mode": mode,
                "step": selected_step,
                "epoch": selected_epoch,
                "runtime_seconds": selected_runtime,
                "selection_value": selected_value,
            }
        )

        for row in group:
            if row.get("step") == selected_step:
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "dataset_label": dataset_label,
                        "method": method,
                        "step": selected_step,
                        "epoch": selected_epoch,
                        "runtime_seconds": selected_runtime,
                        "metric": row.get("metric"),
                        "value": row.get("value"),
                    }
                )

    write_csv(out_dir / "unified_checkpoints_foot.csv", checkpoints)
    write_csv(out_dir / "unified_checkpoint_metrics_foot_long.csv", metric_rows)
    print(f"[OK] foot checkpoints: {len(checkpoints)} runs")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_history_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sfem_run_records(runs_root: Path, runs_master_path: Path | None) -> list[dict[str, Any]]:
    master: dict[str, dict[str, Any]] = {}
    if runs_master_path and runs_master_path.exists():
        for row in load_json(runs_master_path):
            if row.get("project") == "sfem_surrogates":
                master[str(row.get("id"))] = row

    records = []
    for history_path in sorted((runs_root / "sfem_surrogates").glob("*/history.jsonl.gz")):
        run_id = history_path.parent.name
        meta = master.get(run_id, {})
        run_json_path = history_path.parent / "run.json"
        if run_json_path.exists():
            try:
                run_json = load_json(run_json_path)
            except Exception:
                run_json = {}
        else:
            run_json = {}
        records.append(
            {
                "run_id": run_id,
                "method": meta.get("method") or run_json.get("method") or infer_method(meta.get("name") or run_json.get("name") or ""),
                "name": meta.get("name") or run_json.get("name") or run_id,
                "state": meta.get("state") or run_json.get("state") or "",
                "history_path": history_path,
            }
        )
    return records


def infer_method(name: str) -> str:
    lowered = name.lower()
    if "ffn" in lowered:
        return "ffn"
    if "pinn" in lowered:
        return "pinn"
    if "gnn" in lowered or "gno" in lowered:
        return "gno"
    return "unknown"


def reselect_sfem(
    runs_root: Path,
    runs_master_path: Path | None,
    selection_metric: str,
    out_dir: Path,
    state_filter: set[str],
    mode_override: str | None,
) -> None:
    mode = metric_mode(selection_metric, mode_override)
    checkpoints = []
    metric_rows = []

    for record in sfem_run_records(runs_root, runs_master_path):
        if state_filter and str(record.get("state")) not in state_filter:
            continue

        selected_value: float | None = None
        selected_row: dict[str, Any] | None = None
        row_count = 0
        for row in iter_history_rows(Path(record["history_path"])):
            row_count += 1
            value = as_float(row.get(selection_metric))
            if value is None:
                continue
            if better(value, selected_value, mode):
                selected_value = value
                selected_row = row

        if selected_value is None or selected_row is None:
            continue

        step = selected_row.get("_step", selected_row.get("train/epoch/global_step", ""))
        epoch = selected_row.get("train/epoch/epoch", selected_row.get("train/step/epoch", ""))
        runtime = selected_row.get("_runtime", selected_row.get("train/runtime/wall_clock_elapsed_sec", ""))
        checkpoints.append(
            {
                "run_id": record["run_id"],
                "method": record["method"],
                "name": record["name"],
                "state": record["state"],
                "selection_metric": selection_metric,
                "selection_mode": mode,
                "step": step,
                "epoch": epoch,
                "runtime_seconds": runtime,
                "selection_value": selected_value,
                "history_rows_scanned": row_count,
            }
        )

        for metric, value in selected_row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_rows.append(
                    {
                        "run_id": record["run_id"],
                        "method": record["method"],
                        "name": record["name"],
                        "state": record["state"],
                        "step": step,
                        "epoch": epoch,
                        "runtime_seconds": runtime,
                        "metric": metric,
                        "value": value,
                    }
                )

    best_by_method = []
    for method in sorted({row["method"] for row in checkpoints}):
        candidates = [row for row in checkpoints if row["method"] == method]
        best = None
        best_value = None
        for row in candidates:
            value = as_float(row["selection_value"])
            if value is not None and better(value, best_value, mode):
                best_value = value
                best = row
        if best is not None:
            best_by_method.append(best)

    write_csv(out_dir / "unified_checkpoints_sfem_all.csv", checkpoints)
    write_csv(out_dir / "unified_checkpoints_sfem_best_by_method.csv", best_by_method)
    write_csv(out_dir / "unified_checkpoint_metrics_sfem_long.csv", metric_rows)
    print(f"[OK] SFEM checkpoints: {len(checkpoints)} runs, {len(best_by_method)} best-by-method rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospectively reselect report checkpoints.")
    parser.add_argument("--out-dir", type=Path, default=Path("results_visualization/derived/reselected"))
    parser.add_argument("--foot-run-metrics", type=Path, default=Path("results_visualization/derived/run_metrics_long.csv"))
    parser.add_argument("--foot-selection-metric", default="val/pooled/pressure/nrmse")
    parser.add_argument("--skip-foot", action="store_true")
    parser.add_argument("--sfem-runs-root", type=Path, default=Path("wandb_evolution/wandb_evolution_output/runs"))
    parser.add_argument("--runs-master", type=Path, default=Path("wandb_evolution_output/runs_master.json"))
    parser.add_argument("--sfem-selection-metric", default="val/pooled/von_mises/nrmse")
    parser.add_argument("--sfem-selection-mode", choices=["min", "max"], default=None)
    parser.add_argument("--sfem-state", action="append", default=[], help="Optional state filter, e.g. --sfem-state finished")
    parser.add_argument("--skip-sfem", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_foot:
        reselect_foot(
            run_metrics_path=args.foot_run_metrics,
            selection_metric=args.foot_selection_metric,
            out_dir=args.out_dir,
        )
    if not args.skip_sfem:
        reselect_sfem(
            runs_root=args.sfem_runs_root,
            runs_master_path=args.runs_master,
            selection_metric=args.sfem_selection_metric,
            out_dir=args.out_dir,
            state_filter=set(args.sfem_state),
            mode_override=args.sfem_selection_mode,
        )


if __name__ == "__main__":
    main()
