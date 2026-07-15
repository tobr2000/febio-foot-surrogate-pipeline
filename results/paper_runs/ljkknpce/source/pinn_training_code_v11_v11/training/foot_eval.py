from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _denorm(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * std.to(value.device, value.dtype) + mean.to(value.device, value.dtype)


def _metric_bundle(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    err = yp - yt
    mse = torch.mean(err.square())
    rmse = torch.sqrt(mse + eps)
    mae = torch.mean(torch.abs(err))
    maxae = torch.max(torch.abs(err))
    denom = torch.clamp(torch.max(yt) - torch.min(yt), min=eps)
    ss_res = torch.sum(err.square())
    ss_tot = torch.sum((yt - torch.mean(yt)).square())
    r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=eps)
    corr_den = torch.sqrt(torch.sum((yt - yt.mean()).square()) * torch.sum((yp - yp.mean()).square()) + eps)
    pearson = torch.sum((yt - yt.mean()) * (yp - yp.mean())) / corr_den
    return {
        "mse": float(mse.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "maxae": float(maxae.detach().cpu()),
        "nrmse": float((rmse / denom).detach().cpu()),
        "r2": float(r2.detach().cpu()),
        "pearsonr": float(pearson.detach().cpu()),
    }


def _r2_single(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = torch.sum((yp - yt).square())
    ss_tot = torch.sum((yt - torch.mean(yt)).square())
    return float((1.0 - ss_res / torch.clamp(ss_tot, min=eps)).detach().cpu())


def _center_of_pressure(pressure: torch.Tensor, grid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.relu(pressure)
    total = torch.clamp(p.sum(dim=1), min=eps)
    return (p * grid).sum(dim=1) / total


def _empty_metrics() -> dict[str, float]:
    return {
        "mse": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "maxae": float("nan"),
        "nrmse": float("nan"),
        "r2": float("nan"),
        "pearsonr": float("nan"),
    }


def _prediction_diagnostics(y_pred: torch.Tensor) -> dict[str, float]:
    return {
        "pressure_pred_min": float(y_pred.min().detach().cpu()),
        "pressure_pred_max": float(y_pred.max().detach().cpu()),
        "pressure_pred_mean": float(y_pred.mean().detach().cpu()),
        "pressure_pred_std": float(y_pred.std(unbiased=False).detach().cpu()),
        "pressure_pred_negative_fraction": float((y_pred < 0).float().mean().detach().cpu()),
    }


def _build_metric_group(y_true_all: torch.Tensor, y_pred_all: torch.Tensor, grid_all: torch.Tensor) -> dict[str, Any]:
    if y_true_all.numel() == 0:
        return {
            "pressure": _empty_metrics(),
            "reaction_proxy": _empty_metrics(),
            "peak_pressure": _empty_metrics(),
            "center_of_pressure": _empty_metrics(),
            "diagnostics": {},
        }
    true_sum = y_true_all.sum(dim=1)
    pred_sum = y_pred_all.sum(dim=1)
    true_peak = y_true_all.max(dim=1).values
    pred_peak = y_pred_all.max(dim=1).values
    true_cop = _center_of_pressure(y_true_all, grid_all)
    pred_cop = _center_of_pressure(y_pred_all, grid_all)
    return {
        "pressure": _metric_bundle(y_true_all, y_pred_all),
        "reaction_proxy": _metric_bundle(true_sum, pred_sum),
        "peak_pressure": _metric_bundle(true_peak, pred_peak),
        "center_of_pressure": _metric_bundle(true_cop, pred_cop),
        "diagnostics": _prediction_diagnostics(y_pred_all),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    pressure_mean = torch.as_tensor(loader.dataset.stats["pressure_mean"], dtype=torch.float32, device=device)
    pressure_std = torch.as_tensor(loader.dataset.stats["pressure_std"], dtype=torch.float32, device=device)
    pairs = []
    base_id_chunks = []
    sample_metrics = []
    inference_sec = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            t0 = time.perf_counter()
            pred = model(batch)
            inference_sec += time.perf_counter() - t0
            y_true_norm = batch["pressure"]
            y_pred_norm = pred["pressure"]
            y_true = batch.get("pressure_raw", _denorm(y_true_norm, pressure_mean, pressure_std))
            y_pred = _denorm(y_pred_norm, pressure_mean, pressure_std)
            pairs.append((y_true.detach().cpu(), y_pred.detach().cpu(), batch["contact_x"].detach().cpu()))
            base_id_chunks.append(batch["base_model_id"].detach().cpu())

            true_sum = y_true.sum(dim=1)
            pred_sum = y_pred.sum(dim=1)
            true_peak = y_true.max(dim=1).values
            pred_peak = y_pred.max(dim=1).values
            true_cop = _center_of_pressure(y_true, batch["contact_x"])
            pred_cop = _center_of_pressure(y_pred, batch["contact_x"])
            for i, sample_id in enumerate(batch["sample_id"].detach().cpu().tolist()):
                sample_metrics.append(
                    {
                        "sample_id": int(sample_id),
                        "pressure_r2": _r2_single(y_true[i], y_pred[i]),
                        "pressure_rmse": float(torch.sqrt(torch.mean((y_pred[i] - y_true[i]).square())).cpu()),
                        "pressure_mae": float(torch.mean(torch.abs(y_pred[i] - y_true[i])).cpu()),
                        "center_of_pressure_r2": _r2_single(true_cop[i], pred_cop[i]),
                        "reaction_abs_error": float(torch.abs(pred_sum[i] - true_sum[i]).cpu()),
                        "peak_abs_error": float(torch.abs(pred_peak[i] - true_peak[i]).cpu()),
                        "cop_error": float(torch.linalg.norm(pred_cop[i] - true_cop[i]).cpu()),
                    }
                )
            n_batches += 1

    y_true_all = torch.cat([p[0] for p in pairs], dim=0)
    y_pred_all = torch.cat([p[1] for p in pairs], dim=0)
    grid_all = torch.cat([p[2] for p in pairs], dim=0)
    base_ids_all = torch.cat(base_id_chunks, dim=0)

    pooled = _build_metric_group(y_true_all, y_pred_all, grid_all)
    seen_mask = base_ids_all < 10
    unseen_mask = base_ids_all >= 10
    groups = {}
    if bool(seen_mask.any()):
        groups["seen_bases_00_09"] = _build_metric_group(y_true_all[seen_mask], y_pred_all[seen_mask], grid_all[seen_mask])
    if bool(unseen_mask.any()):
        groups["unseen_bases_10_11"] = _build_metric_group(y_true_all[unseen_mask], y_pred_all[unseen_mask], grid_all[unseen_mask])
    by_base = {}
    for base_id in sorted(set(int(x) for x in base_ids_all.tolist())):
        mask = base_ids_all == base_id
        by_base[f"base_{base_id:02d}"] = _build_metric_group(y_true_all[mask], y_pred_all[mask], grid_all[mask])

    result = {
        "pooled": pooled,
        "groups": groups,
        "by_base": by_base,
        "per_sample_summary": summarize_sample_metrics(sample_metrics),
        "per_sample": sample_metrics,
        "timing": {
            "inference_sec": inference_sec,
            "mean_inference_sec_per_batch": inference_sec / max(1, n_batches),
            "n_batches": n_batches,
            "n_samples": int(y_true_all.shape[0]),
            "n_contact_faces": int(y_true_all.shape[1]),
        },
    }
    if was_training:
        model.train()
    return result


def summarize_sample_metrics(sample_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not sample_metrics:
        return out
    keys = [key for key in sample_metrics[0].keys() if key != "sample_id"]
    for key in keys:
        vals = np.asarray([row[key] for row in sample_metrics], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p10": float(np.percentile(vals, 10)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
        }
    return out


def flatten_metrics(result: dict[str, Any], prefix: str = "") -> dict[str, float]:
    p = f"{prefix}/" if prefix else ""
    flat: dict[str, float] = {}
    for target, metrics in result.get("pooled", {}).items():
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            flat[f"{p}pooled/{target}/{name}"] = float(value)
    for group_name, group in result.get("groups", {}).items():
        for target, metrics in group.items():
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                flat[f"{p}{group_name}/{target}/{name}"] = float(value)
    for base_name, group in result.get("by_base", {}).items():
        for target, metrics in group.items():
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                flat[f"{p}by_base/{base_name}/{target}/{name}"] = float(value)
    for metric, stats in result.get("per_sample_summary", {}).items():
        for name, value in stats.items():
            flat[f"{p}per_sample/{metric}/{name}"] = float(value)
    for name, value in result.get("timing", {}).items():
        flat[f"{p}timing/{name}"] = float(value)
    return flat


def print_eval_summary(result: dict[str, Any], title: str = "eval") -> None:
    pressure = result["pooled"]["pressure"]
    reaction = result["pooled"]["reaction_proxy"]
    peak = result["pooled"]["peak_pressure"]
    cop = result["pooled"]["center_of_pressure"]
    print(f"\n[EVAL] {title}")
    print(
        f"  pressure: R2={pressure['r2']:.6f} RMSE={pressure['rmse']:.6f} "
        f"MAE={pressure['mae']:.6f} NRMSE={pressure['nrmse']:.6f}"
    )
    pressure_r2_summary = result.get("per_sample_summary", {}).get("pressure_r2")
    if pressure_r2_summary is not None:
        print(
            "  pressure per-sample R2: "
            f"p10={pressure_r2_summary['p10']:.6f} "
            f"p50={pressure_r2_summary['p50']:.6f} "
            f"p90={pressure_r2_summary['p90']:.6f}"
        )
    print(f"  reaction proxy: RMSE={reaction['rmse']:.6f} MAE={reaction['mae']:.6f}")
    print(f"  peak pressure: RMSE={peak['rmse']:.6f} MAE={peak['mae']:.6f}")
    print(f"  center of pressure: RMSE={cop['rmse']:.6f} MAE={cop['mae']:.6f}")


def save_eval_report(result: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
