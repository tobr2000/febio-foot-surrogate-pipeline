from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import BatchSampler, DataLoader

from foot_data import PARAM_NAMES, FootShardDataset, compute_stats, foot_collate
from foot_eval import evaluate_model, flatten_metrics, print_eval_summary, save_eval_report

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
            fcntl.flock(handle, fcntl.LOCK_UN)
        except ImportError:
            yield


def ensure_stats(cfg: dict[str, Any]) -> dict[str, Any]:
    stats_path = Path(cfg["data"]["stats_path"])
    required = {"pressure_mean", "node_pos_mean", "node_disp_mean"}
    lock_path = stats_path.with_suffix(stats_path.suffix + ".lock")
    print(f"[STARTUP] stats_path={stats_path}", flush=True)
    with file_lock(lock_path):
        print(f"[STARTUP] acquired_stats_lock={lock_path}", flush=True)
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                if required.issubset(stats.keys()) and stats.get("param_names") == PARAM_NAMES:
                    print(
                        f"[STARTUP] loaded_existing_stats sample_count={stats.get('stats_sample_count')}",
                        flush=True,
                    )
                    return stats
                print(f"[WARN] Stats file {stats_path} is missing required keys; recomputing.", flush=True)
            except json.JSONDecodeError:
                print(f"[WARN] Stats file {stats_path} is incomplete/corrupt; recomputing.", flush=True)
        print(
            f"[STARTUP] computing_stats shard_dir={cfg['data']['shard_dir']} "
            f"max_samples={cfg['data'].get('stats_max_samples', 2000)}",
            flush=True,
        )
        stats = compute_stats(
            cfg["data"]["shard_dir"],
            out_path=stats_path,
            max_samples=cfg["data"].get("stats_max_samples", 2000),
        )
        print(f"[STARTUP] computed_stats sample_count={stats.get('stats_sample_count')}", flush=True)
        return stats


def resolve_checkpoint_dir(cfg: dict[str, Any], model_name: str) -> Path:
    root = Path(cfg["training"]["checkpoint_dir"])
    suffix = os.environ.get("RUN_NAME_SUFFIX", "").strip()
    run_name = f"{model_name}_{suffix}" if suffix else model_name
    return root / run_name


def dataset_fingerprint(shard_dir: str | Path) -> dict[str, Any]:
    root = Path(shard_dir)
    paths = sorted(root.glob("batch_*.npz"))
    sidecar_paths = sorted(root.glob("batch_*_history.npy"))
    metadata_paths = [p for p in (root / "modelready_metadata.json", root / "pinn_history_manifest.json") if p.exists()]
    h = hashlib.sha256()
    total_size = 0
    total_samples = 0
    dataset_ids: set[str] = set()
    sample_min: int | None = None
    sample_max: int | None = None
    for path in paths:
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        total_size += stat.st_size
        h.update(f"{rel}|{stat.st_size}|{int(stat.st_mtime_ns)}\n".encode("utf-8"))
        try:
            with np.load(path, allow_pickle=False) as data:
                sample_ids = data["sample_ids"].astype(int)
                if "dataset_ids" in data.files:
                    dataset_ids.update(str(value) for value in data["dataset_ids"].astype(str).tolist())
                total_samples += int(sample_ids.size)
                if sample_ids.size:
                    local_min = int(sample_ids.min())
                    local_max = int(sample_ids.max())
                    sample_min = local_min if sample_min is None else min(sample_min, local_min)
                    sample_max = local_max if sample_max is None else max(sample_max, local_max)
                    h.update(f"{rel}|samples|{sample_ids.size}|{local_min}|{local_max}\n".encode("utf-8"))
        except Exception as exc:
            h.update(f"{rel}|sample_ids_unavailable|{type(exc).__name__}\n".encode("utf-8"))
    for path in sidecar_paths + metadata_paths:
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        total_size += stat.st_size
        h.update(f"{rel}|{stat.st_size}|{int(stat.st_mtime_ns)}\n".encode("utf-8"))
    return {
        "shard_dir": str(root),
        "shard_count": len(paths),
        "history_sidecar_count": len(sidecar_paths),
        "metadata_file_count": len(metadata_paths),
        "packed_sample_count": total_samples,
        "sample_id_min": sample_min,
        "sample_id_max": sample_max,
        "dataset_ids": sorted(dataset_ids),
        "total_size_bytes": total_size,
        "metadata_sha256": h.hexdigest(),
    }


def write_run_manifest(run_dir: Path, cfg: dict[str, Any], stats: dict[str, Any]) -> Path:
    manifest = {
        "config": cfg,
        "stats_sample_count": stats.get("stats_sample_count"),
        "param_names": stats.get("param_names"),
        "created_unix_time": time.time(),
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def log_run_artifacts(wb_run: Any, model_name: str, run_dir: Path, cfg: dict[str, Any], stats_path: Path) -> None:
    try:
        import wandb
    except Exception:
        return
    code_artifact = wandb.Artifact(f"{model_name}_training_code", type="code")
    code_paths = [
        "training/foot_config.yaml",
        "training/train_common.py",
        "training/foot_data.py",
        "training/foot_eval.py",
        "training/model_gno.py",
        "training/model_ffn.py",
        "training/model_pinn.py",
        "training/models_common.py",
        "training/train_gno.py",
        "training/train_ffn.py",
        "training/train_pinn.py",
        "training/run_training.slurm",
        "training/requirements-training.txt",
        "training/analyze_dataset_quality.py",
        "scripts/generate_manifest.py",
        "scripts/generate_base_templates.py",
        "scripts/run_batch.py",
        "scripts/pack_batch.py",
        "scripts/common.py",
    ]
    for text_path in code_paths:
        path = Path(text_path)
        if path.exists():
            code_artifact.add_file(str(path), name=text_path)
    wb_run.log_artifact(code_artifact)

    data_artifact = wandb.Artifact(f"{model_name}_dataset_fingerprint", type="dataset")
    manifest_path = write_run_manifest(run_dir, cfg, json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {})
    data_artifact.add_file(str(manifest_path), name="run_manifest.json")
    for text_path in [
        stats_path,
        Path("training/dataset_quality/summary.json"),
        Path("training/dataset_quality/regression_baselines.json"),
        Path("training/dataset_quality/valid_counts_by_base.csv"),
    ]:
        if Path(text_path).exists():
            data_artifact.add_file(str(text_path), name=str(text_path).replace("\\", "/"))
    data_artifact.metadata = cfg.get("data", {})
    wb_run.log_artifact(data_artifact)


def make_loaders(cfg: dict[str, Any], stats: dict[str, Any]) -> tuple[FootShardDataset, FootShardDataset, DataLoader, DataLoader]:
    tcfg = cfg["training"]
    include_history = str(tcfg.get("model", cfg.get("training", {}).get("model", ""))).lower() == "pinn"
    train_ds = FootShardDataset(
        cfg["data"]["shard_dir"],
        split="train",
        stats=stats,
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["data"]["seed"]),
        max_samples=cfg["data"].get("train_max_samples"),
        max_open_shards=int(tcfg.get("max_open_shards", 4)),
        train_base_ids=cfg["data"].get("train_base_ids"),
        validation_base_ids=cfg["data"].get("validation_base_ids"),
        include_all_unseen_base_ids_in_val=bool(cfg["data"].get("include_all_unseen_base_ids_in_val", True)),
        include_history=include_history,
        history_time_samples=tcfg.get("history_time_samples"),
        node_history_point_samples=tcfg.get("node_history_point_samples"),
        element_history_point_samples=tcfg.get("element_history_point_samples"),
        contact_history_point_samples=tcfg.get("contact_history_point_samples"),
    )
    val_ds = FootShardDataset(
        cfg["data"]["shard_dir"],
        split="val",
        stats=stats,
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["data"]["seed"]),
        max_samples=cfg["data"].get("val_max_samples"),
        max_open_shards=int(tcfg.get("max_open_shards", 4)),
        train_base_ids=cfg["data"].get("train_base_ids"),
        validation_base_ids=cfg["data"].get("validation_base_ids"),
        include_all_unseen_base_ids_in_val=bool(cfg["data"].get("include_all_unseen_base_ids_in_val", True)),
        include_history=include_history,
        history_time_samples=tcfg.get("val_history_time_samples", tcfg.get("history_time_samples")),
        node_history_point_samples=tcfg.get("val_node_history_point_samples", tcfg.get("node_history_point_samples")),
        element_history_point_samples=tcfg.get("val_element_history_point_samples", tcfg.get("element_history_point_samples")),
        contact_history_point_samples=tcfg.get("val_contact_history_point_samples", tcfg.get("contact_history_point_samples")),
    )
    device = choose_device(str(cfg["training"]["device"]))
    num_workers = int(tcfg["num_workers"])
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(tcfg.get("persistent_workers", False)) and num_workers > 0,
    }
    if include_history:
        loader_kwargs["collate_fn"] = foot_collate
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(tcfg.get("prefetch_factor", 2))
    batch_size = int(cfg["training"]["batch_size"])
    if bool(tcfg.get("shard_aware_batches", True)):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ShardAwareBatchSampler(
            train_ds,
            batch_size=batch_size,
            seed=int(cfg["data"]["seed"]),
            drop_last=False,
            balance_base_ids=bool(tcfg.get("balance_base_ids_in_batch", True)),
        ),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_ds, val_ds, train_loader, val_loader


class ShardAwareBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: FootShardDataset,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
        balance_base_ids: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.balance_base_ids = bool(balance_base_ids)
        groups: dict[Path, list[int]] = {}
        for idx, ref in enumerate(dataset.refs):
            groups.setdefault(ref.shard_path, []).append(idx)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shard_order = rng.permutation(len(self.groups))
        for shard_idx in shard_order:
            indices = np.asarray(self.groups[int(shard_idx)], dtype=np.int64)
            if not self.balance_base_ids:
                rng.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start : start + self.batch_size].tolist()
                    if len(batch) == self.batch_size or not self.drop_last:
                        yield batch
                continue

            by_base: dict[int, list[int]] = {}
            for idx in indices.tolist():
                base_id = self.dataset.refs[int(idx)].base_model_id
                by_base.setdefault(base_id, []).append(int(idx))
            for values in by_base.values():
                rng.shuffle(values)
            base_ids = list(by_base)
            rng.shuffle(base_ids)
            cursors = {base_id: 0 for base_id in base_ids}
            while True:
                active = [base_id for base_id in base_ids if cursors[base_id] < len(by_base[base_id])]
                if not active:
                    break
                batch: list[int] = []
                while active and len(batch) < self.batch_size:
                    for base_id in list(active):
                        if cursors[base_id] < len(by_base[base_id]):
                            batch.append(by_base[base_id][cursors[base_id]])
                            cursors[base_id] += 1
                            if len(batch) >= self.batch_size:
                                break
                    active = [base_id for base_id in base_ids if cursors[base_id] < len(by_base[base_id])]
                rng.shuffle(batch)
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
        self.seed += 1

    def __len__(self) -> int:
        total = 0
        for group in self.groups:
            if self.drop_last:
                total += len(group) // self.batch_size
            else:
                total += int(np.ceil(len(group) / self.batch_size))
        return total


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


class StepTimer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values: dict[str, list[float]] = {
            "data_wait_sec": [],
            "to_device_sec": [],
            "forward_sec": [],
            "loss_sec": [],
            "backward_sec": [],
            "optimizer_sec": [],
            "step_total_sec": [],
        }

    def add(self, key: str, value: float) -> None:
        self.values[key].append(float(value))

    def summary(self) -> dict[str, float]:
        return {
            f"timing/{key}_mean_100": float(np.mean(values)) if values else 0.0
            for key, values in self.values.items()
        }


def stat_tensor(stats: dict[str, Any], key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(stats[key], device=device, dtype=dtype)


def denorm(value: torch.Tensor, stats: dict[str, Any], mean_key: str, std_key: str) -> torch.Tensor:
    mean = stat_tensor(stats, mean_key, value.device, value.dtype)
    std = stat_tensor(stats, std_key, value.device, value.dtype)
    return value * std + mean


def contact_region_features(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    contact_x = batch.get("contact_x")
    if contact_x is None or contact_x.shape[-1] < 7:
        return None
    geom_x = batch.get("contact_geom_x")
    if geom_x is not None and contact_x.shape[-1] >= geom_x.shape[-1] + 4:
        region = contact_x[..., geom_x.shape[-1] : geom_x.shape[-1] + 4]
    else:
        region = contact_x[..., -4:]
    if region.shape[-1] != 4:
        return None
    return region


def normalized_physical_components(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    lcfg = cfg["loss"]
    y = batch["pressure"]
    p = pred["pressure"]
    p_raw = denorm(p, stats, "pressure_mean", "pressure_std")
    y_raw = batch["pressure_raw"]
    p_pos = torch.relu(p_raw)
    pressure = torch.mean((p - y).square())
    true_sum = y_raw.sum(dim=1)
    pred_sum = p_pos.sum(dim=1)
    reaction_scale = torch.clamp(true_sum.abs(), min=1.0)
    reaction = torch.mean(((pred_sum - true_sum) / reaction_scale).square())
    true_peak = y_raw.max(dim=1).values
    pred_peak = p_pos.max(dim=1).values
    peak_scale = torch.clamp(true_peak.abs(), min=0.05)
    peak = torch.mean(((pred_peak - true_peak) / peak_scale).square())
    nonnegative = torch.mean(torch.relu(-p_raw).square())
    components = {
        "pressure": pressure * float(lcfg["pressure_weight"]),
        "reaction": reaction * float(lcfg["reaction_weight"]),
        "peak": peak * float(lcfg["peak_weight"]),
        "nonnegative_pressure": nonnegative * float(lcfg.get("nonnegative_pressure_weight", 0.02)),
    }
    topk_weight = float(lcfg.get("topk_pressure_weight", 0.0))
    if topk_weight > 0.0:
        frac = float(lcfg.get("topk_pressure_fraction", 0.05))
        k = max(1, int(round(y.shape[1] * frac)))
        top_idx = torch.topk(y_raw.squeeze(-1), k=k, dim=1).indices
        top_pred = torch.gather(p, 1, top_idx.unsqueeze(-1))
        top_true = torch.gather(y, 1, top_idx.unsqueeze(-1))
        components["topk_pressure"] = torch.mean((top_pred - top_true).square()) * topk_weight

    region = contact_region_features(batch)
    if region is not None:
        true_region_sum = torch.sum(y_raw * region, dim=1)
        pred_region_sum = torch.sum(p_pos * region, dim=1)
        region_scale = torch.clamp(true_region_sum.abs(), min=0.1)
        components["regional_reaction"] = torch.mean(
            ((pred_region_sum - true_region_sum) / region_scale).square()
        ) * float(lcfg.get("regional_reaction_weight", 0.0))

        true_region_peak = []
        pred_region_peak = []
        valid_region = []
        y_scalar = y_raw.squeeze(-1)
        p_scalar = p_pos.squeeze(-1)
        for ridx in range(region.shape[-1]):
            mask = region[..., ridx] > 0.5
            valid = mask.any(dim=1)
            true_masked = torch.where(mask, y_scalar, torch.full_like(y_scalar, -1.0e9))
            pred_masked = torch.where(mask, p_scalar, torch.full_like(p_scalar, -1.0e9))
            true_region_peak.append(true_masked.max(dim=1).values)
            pred_region_peak.append(pred_masked.max(dim=1).values)
            valid_region.append(valid)
        true_region_peak_t = torch.stack(true_region_peak, dim=1)
        pred_region_peak_t = torch.stack(pred_region_peak, dim=1)
        valid_region_t = torch.stack(valid_region, dim=1).to(device=y_raw.device, dtype=y_raw.dtype)
        peak_region_scale = torch.clamp(true_region_peak_t.abs(), min=0.05)
        region_peak_error = ((pred_region_peak_t - true_region_peak_t) / peak_region_scale).square()
        components["regional_peak"] = (
            torch.sum(region_peak_error * valid_region_t) / torch.clamp(valid_region_t.sum(), min=1.0)
        ) * float(lcfg.get("regional_peak_weight", 0.0))

    if "reaction_mean_norm" in pred:
        true_reaction_mean_norm = y.mean(dim=1).mean(dim=-1, keepdim=True)
        components["reaction_head"] = torch.mean(
            (pred["reaction_mean_norm"] - true_reaction_mean_norm).square()
        ) * float(lcfg.get("reaction_head_weight", 0.0))
    if "peak_norm" in pred:
        true_peak_norm = y.max(dim=1).values
        components["peak_head"] = torch.mean((pred["peak_norm"] - true_peak_norm).square()) * float(
            lcfg.get("peak_head_weight", 0.0)
        )
    if "pressure_shape" in pred:
        components["pressure_shape_anchor"] = torch.mean((pred["pressure_shape"] - y).square()) * float(
            lcfg.get("pressure_shape_anchor_weight", 0.0)
        )
    if "pressure_residual" in pred:
        components["pressure_residual_l2"] = torch.mean(pred["pressure_residual"].square()) * float(
            lcfg.get("pressure_residual_l2_weight", 0.0)
        )
    return components


def masked_mean_square(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    squared = value.square()
    if mask is None:
        return torch.mean(squared)
    while mask.ndim < squared.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=squared.device, dtype=squared.dtype)
    denom = torch.clamp(torch.sum(torch.ones_like(squared) * mask), min=1.0)
    return torch.sum(squared * mask) / denom


def base_pressure_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    components = normalized_physical_components(pred, batch, cfg, stats)
    loss = sum(components.values())
    return loss, {key: float(value.detach().cpu()) for key, value in components.items()}, components


def ramp_multiplier(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - start_epoch + 1) / ramp_epochs)))


def _component_cadence_active(name: str, cfg: dict[str, Any], global_step: int | None) -> bool:
    if global_step is None:
        return True
    pcfg = cfg.get("pinn_curriculum", {})
    if name in {"pressure", "reaction", "peak", "nonnegative_pressure", "topk_pressure", "regional_reaction", "regional_peak", "reaction_head", "peak_head", "pressure_shape_anchor", "pressure_residual_l2"}:
        every = 1
    elif name in {"element_aux", "node_displacement_aux"}:
        every = int(pcfg.get("data_aux_every_steps", 1))
    elif name.startswith("history_"):
        every = int(pcfg.get("history_every_steps", pcfg.get("physics_every_steps", 1)))
    elif name in {"contact_projection", "contact_complementarity"}:
        every = int(pcfg.get("contact_every_steps", pcfg.get("physics_every_steps", 1)))
    else:
        every = int(pcfg.get("physics_every_steps", 1))
    return every <= 1 or global_step % every == 0


def pinn_component_multiplier(name: str, cfg: dict[str, Any], epoch: int, global_step: int | None = None) -> float:
    if not _component_cadence_active(name, cfg, global_step):
        return 0.0
    pcfg = cfg.get("pinn_curriculum", {})
    if name in {"pressure", "reaction", "peak", "nonnegative_pressure", "topk_pressure", "regional_reaction", "regional_peak", "reaction_head", "peak_head", "pressure_shape_anchor", "pressure_residual_l2"}:
        return 1.0
    if name in {"element_aux", "node_displacement_aux"}:
        return ramp_multiplier(
            epoch,
            int(pcfg.get("data_aux_start_epoch", 1)),
            int(pcfg.get("data_aux_ramp_epochs", 1)),
        )
    return ramp_multiplier(
        epoch,
        int(pcfg.get("physics_start_epoch", 10)),
        int(pcfg.get("physics_ramp_epochs", 20)),
    )


def isotropic_constitutive_stress(displacement: torch.Tensor, coords: torch.Tensor, params_raw: torch.Tensor) -> torch.Tensor:
    grads = []
    for comp in range(3):
        grad = torch.autograd.grad(
            displacement[:, :, comp].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grads.append(grad)
    grad_u = torch.stack(grads, dim=-2)
    strain = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    e_idx = PARAM_NAMES.index("E_flesh")
    e = params_raw[:, None, e_idx].to(displacement.device, displacement.dtype).clamp_min(0.05)
    nu = torch.full_like(e, 0.45)
    mu = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    trace = strain.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    stress_matrix = 2.0 * mu[:, :, None, None] * strain
    eye = torch.eye(3, device=displacement.device, dtype=displacement.dtype)[None, None, :, :]
    stress_matrix = stress_matrix + lam[:, :, None, None] * trace[:, :, None, None] * eye
    return torch.stack(
        [
            stress_matrix[:, :, 0, 0],
            stress_matrix[:, :, 1, 1],
            stress_matrix[:, :, 2, 2],
            stress_matrix[:, :, 0, 1],
            stress_matrix[:, :, 1, 2],
            stress_matrix[:, :, 0, 2],
        ],
        dim=-1,
    )


def pinn_loss_components(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    stats: dict[str, Any],
    epoch: int,
    global_step: int | None = None,
) -> dict[str, torch.Tensor]:
    lcfg = cfg["loss"]
    parts = normalized_physical_components(pred, batch, cfg, stats)

    if "stress" in pred:
        stress_target = batch["element_y"][:, :, :6]
        vm_target = batch["element_y"][:, :, 6:7]
        stress = pred["stress"]
        vm = pred["von_mises"]
        parts["element_aux"] = (
            torch.mean((stress - stress_target).square()) + torch.mean((vm - vm_target).square())
        ) * float(lcfg["element_aux_weight"])
        parts["vm_consistency"] = torch.mean((vm - pred["von_mises_from_stress"]).square()) * float(
            lcfg["pinn_vm_consistency_weight"]
        )

        coords = batch["element_pos"]
        grads = []
        for comp in range(6):
            grad = torch.autograd.grad(
                stress[:, :, comp].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            grads.append(grad)
        div_x = grads[0][:, :, 0] + grads[3][:, :, 1] + grads[5][:, :, 2]
        div_y = grads[3][:, :, 0] + grads[1][:, :, 1] + grads[4][:, :, 2]
        div_z = grads[5][:, :, 0] + grads[4][:, :, 1] + grads[2][:, :, 2]
        parts["equilibrium"] = torch.mean(div_x.square() + div_y.square() + div_z.square()) * float(
            lcfg["pinn_equilibrium_weight"]
        )

        constitutive = isotropic_constitutive_stress(pred["displacement"], coords, batch["params_raw"])
        constitutive_norm = (constitutive - stat_tensor(stats, "stress_mean", stress.device, stress.dtype)) / stat_tensor(
            stats, "stress_std", stress.device, stress.dtype
        )
        parts["constitutive"] = torch.mean((stress - constitutive_norm).square()) * float(
            lcfg.get("pinn_constitutive_weight", 0.02)
        )

    if "history_stress" in pred and "element_history_y" in batch:
        hist_target = batch["element_history_y"]
        hist_stress_target = hist_target[:, :, :, :6]
        hist_vm_target = hist_target[:, :, :, 6:7]
        hist_mask = batch.get("element_history_valid_mask")
        parts["history_element_aux"] = (
            masked_mean_square(pred["history_stress"] - hist_stress_target, hist_mask)
            + masked_mean_square(pred["history_von_mises"] - hist_vm_target, hist_mask)
        ) * float(lcfg.get("pinn_history_element_aux_weight", lcfg["element_aux_weight"] * 0.5))
        parts["history_vm_consistency"] = masked_mean_square(
            pred["history_von_mises"] - pred["history_von_mises_from_stress"], hist_mask
        ) * float(lcfg.get("pinn_history_vm_consistency_weight", lcfg["pinn_vm_consistency_weight"] * 0.5))

    if "node_displacement" in pred:
        parts["node_displacement_aux"] = torch.mean((pred["node_displacement"] - batch["node_disp"]).square()) * float(
            lcfg["node_aux_weight"]
        )
    if "history_node_displacement" in pred and "node_history_disp" in batch:
        parts["history_node_aux"] = masked_mean_square(
            pred["history_node_displacement"] - batch["node_history_disp"],
            batch.get("node_history_valid_mask"),
        ) * float(lcfg.get("pinn_history_node_aux_weight", lcfg["node_aux_weight"] * 0.5))

    p_raw = denorm(pred["pressure"], stats, "pressure_mean", "pressure_std")
    p_pos = torch.relu(p_raw)
    y_raw = batch["pressure_raw"]
    target_sum = y_raw.sum(dim=1)
    pred_sum = p_pos.sum(dim=1)
    parts["contact_projection"] = torch.mean(((pred_sum - target_sum) / torch.clamp(target_sum.abs(), min=1.0)).square()) * float(
        lcfg["pinn_contact_projection_weight"]
    )
    gap = batch["contact_y_raw"][:, :, 0:1]
    parts["contact_complementarity"] = torch.mean((p_pos * torch.relu(gap)).square()) * float(
        lcfg.get("pinn_contact_complementarity_weight", 0.02)
    )
    if "history_pressure" in pred and "contact_history_y" in batch:
        parts["history_contact_pressure"] = masked_mean_square(
            pred["history_pressure"] - batch["contact_history_y"][:, :, :, 1:2],
            batch.get("contact_history_valid_mask"),
        ) * float(lcfg.get("pinn_history_contact_weight", lcfg["pressure_weight"] * 0.25))
    active_parts: dict[str, torch.Tensor] = {}
    for key, value in parts.items():
        multiplier = pinn_component_multiplier(key, cfg, epoch, global_step)
        if multiplier > 0.0:
            active_parts[key] = value * multiplier
    return active_parts


def _metric_path(result: dict[str, Any], path: str, default: float = float("nan")) -> float:
    cur: Any = result
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def checkpoint_score(result: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Lower-is-better validation score used for checkpoint selection.

    Pressure MSE is still available, but the composite metric protects us from
    choosing a checkpoint that is good on dense pressure while drifting badly on
    peak, reaction, or center-of-pressure behavior.
    """
    ccfg = cfg["training"].get("checkpoint_selection", {})
    metric = str(ccfg.get("metric", "pressure_mse")).lower()
    if metric == "pressure_mse":
        return _metric_path(result, "pooled/pressure/mse", float("inf"))
    group = str(ccfg.get("group", "unseen_bases_10_11"))
    root = result.get("groups", {}).get(group, result.get("pooled", {}))
    if metric == "composite":
        pressure = _metric_path(root, "pressure/nrmse", float("inf"))
        peak = _metric_path(root, "peak_pressure/nrmse", pressure)
        reaction = _metric_path(root, "reaction_proxy/nrmse", pressure)
        cop = _metric_path(root, "center_of_pressure/nrmse", pressure)
        weights = ccfg.get("weights", {})
        return (
            float(weights.get("pressure", 1.0)) * pressure
            + float(weights.get("peak", 0.75)) * peak
            + float(weights.get("reaction", 0.5)) * reaction
            + float(weights.get("center_of_pressure", 0.5)) * cop
        )
    return _metric_path(result, str(ccfg.get("path", "pooled/pressure/mse")), float("inf"))


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        state = model.state_dict()
        backup = {key: state[key].detach().clone() for key in self.shadow if key in state}
        try:
            for key, value in self.shadow.items():
                if key in state:
                    state[key].copy_(value)
            yield
        finally:
            for key, value in backup.items():
                state[key].copy_(value)


class RobALRSLossBalancer:
    """Robust adaptive loss reweighting for multi-residual PINN objectives.

    This is a practical RobALRS-style implementation: it tracks an EMA of each
    unweighted component, assigns inverse-magnitude weights with clipping, and
    renormalizes so active terms keep a stable total scale.
    """

    def __init__(
        self,
        enabled: bool = True,
        ema: float = 0.98,
        min_weight: float = 0.05,
        max_weight: float = 20.0,
        warmup_steps: int = 100,
        eps: float = 1e-8,
        min_weight_by_component: dict[str, float] | None = None,
        max_weight_by_component: dict[str, float] | None = None,
    ) -> None:
        self.enabled = enabled
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_weight_by_component = min_weight_by_component or {}
        self.max_weight_by_component = max_weight_by_component or {}
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.step = 0
        self.loss_ema: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def _clip_weight(self, key: str, value: float) -> float:
        lo = float(self.min_weight_by_component.get(key, self.min_weight))
        hi = float(self.max_weight_by_component.get(key, self.max_weight))
        return float(np.clip(value, lo, hi))

    def combine(self, components: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        self.step += 1
        values = {key: float(value.detach().abs().cpu()) for key, value in components.items()}
        for key, value in values.items():
            if key not in self.loss_ema:
                self.loss_ema[key] = max(value, self.eps)
            else:
                self.loss_ema[key] = self.ema * self.loss_ema[key] + (1.0 - self.ema) * max(value, self.eps)
        if not self.enabled or self.step <= self.warmup_steps:
            self.weights = {key: 1.0 for key in components}
        else:
            raw = {key: 1.0 / max(self.loss_ema[key], self.eps) for key in components}
            mean_raw = sum(raw.values()) / max(1, len(raw))
            self.weights = {key: self._clip_weight(key, raw[key] / max(mean_raw, self.eps)) for key in components}
            mean_weight = sum(self.weights.values()) / max(1, len(self.weights))
            self.weights = {key: self._clip_weight(key, value / max(mean_weight, self.eps)) for key, value in self.weights.items()}
        loss = sum(components[key] * self.weights.get(key, 1.0) for key in components)
        diagnostics: dict[str, float] = {}
        for key, value in components.items():
            diagnostics[key] = float(value.detach().cpu())
            diagnostics[f"robalrs_weight/{key}"] = float(self.weights.get(key, 1.0))
            diagnostics[f"robalrs_ema/{key}"] = float(self.loss_ema.get(key, 0.0))
        return loss, diagnostics


def init_wandb(model_name: str, cfg: dict[str, Any], run_dir: Path) -> Any | None:
    wcfg = cfg.get("wandb", {})
    if not bool(wcfg.get("enabled", False)):
        return None
    try:
        import wandb
    except Exception:
        print("[WARN] wandb is not installed; continuing without W&B.")
        return None
    tags = list(wcfg.get("tags", [])) + [f"model:{model_name}"]
    return wandb.init(
        project=wcfg.get("project", os.environ.get("WANDB_PROJECT", "vt2-febio-foot")),
        entity=wcfg.get("entity", None) or os.environ.get("WANDB_ENTITY"),
        mode=wcfg.get("mode", os.environ.get("WANDB_MODE", "online")),
        group=wcfg.get("group", "contact-pressure"),
        name=f"{model_name}_contact_pressure",
        tags=tags,
        config=cfg,
        dir=str(run_dir),
        reinit=True,
    )


def run_training(model_name: str, build_model_fn, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/foot_config.yaml")
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--timing-log-every", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["training"]["model"] = model_name
    model_override = cfg.get("model_overrides", {}).get(model_name, {})
    if model_override:
        deep_update(cfg, model_override)
    if args.shard_dir:
        cfg["data"]["shard_dir"] = args.shard_dir
    stats_env = os.environ.get("STATS_PATH", "").strip()
    default_stats = str(cfg["data"].get("stats_path", "training/foot_stats.json"))
    if stats_env:
        cfg["data"]["stats_path"] = stats_env
    elif args.shard_dir and default_stats == "training/foot_stats.json":
        shard_name = Path(args.shard_dir).name or "default"
        cfg["data"]["stats_path"] = f"training/foot_stats_{shard_name}.json"
    stats_max_env = os.environ.get("STATS_MAX_SAMPLES", "").strip()
    if stats_max_env:
        cfg["data"]["stats_max_samples"] = int(stats_max_env)
    sampling_env = {
        "HISTORY_TIME_SAMPLES": "history_time_samples",
        "NODE_HISTORY_POINT_SAMPLES": "node_history_point_samples",
        "ELEMENT_HISTORY_POINT_SAMPLES": "element_history_point_samples",
        "CONTACT_HISTORY_POINT_SAMPLES": "contact_history_point_samples",
        "VAL_HISTORY_TIME_SAMPLES": "val_history_time_samples",
        "VAL_NODE_HISTORY_POINT_SAMPLES": "val_node_history_point_samples",
        "VAL_ELEMENT_HISTORY_POINT_SAMPLES": "val_element_history_point_samples",
        "VAL_CONTACT_HISTORY_POINT_SAMPLES": "val_contact_history_point_samples",
    }
    for env_name, cfg_name in sampling_env.items():
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            cfg["training"][cfg_name] = int(env_value)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.timing_log_every is not None:
        cfg["training"]["timing_log_every"] = args.timing_log_every
    if args.max_samples:
        cfg["data"]["train_max_samples"] = args.max_samples
        cfg["data"]["val_max_samples"] = max(1, args.max_samples // 5)
    if args.wandb_mode:
        cfg.setdefault("wandb", {})["enabled"] = args.wandb_mode != "disabled"
        cfg["wandb"]["mode"] = args.wandb_mode if args.wandb_mode != "disabled" else "offline"

    print(
        f"[STARTUP] model={model_name} shard_dir={cfg['data']['shard_dir']} "
        f"batch_size={cfg['training']['batch_size']} num_workers={cfg['training']['num_workers']}",
        flush=True,
    )
    set_seed(int(cfg["data"]["seed"]))
    print("[STARTUP] fingerprinting_dataset", flush=True)
    cfg["data"]["fingerprint"] = dataset_fingerprint(cfg["data"]["shard_dir"])
    print(f"[STARTUP] fingerprint={cfg['data']['fingerprint']}", flush=True)
    stats = ensure_stats(cfg)
    print("[STARTUP] building_datasets_and_loaders", flush=True)
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, stats)
    print(f"[STARTUP] train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    device = choose_device(str(cfg["training"]["device"]))
    print(f"[STARTUP] device={device}", flush=True)
    sample = train_ds[0]
    print("[STARTUP] building_model", flush=True)
    model = build_model_fn(len(PARAM_NAMES), cfg, sample).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler_name = str(cfg["training"].get("lr_scheduler", "cosine")).lower()
    scheduler: Any
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg["training"].get("lr_plateau_factor", 0.5)),
            patience=int(cfg["training"].get("lr_plateau_patience", 15)),
            min_lr=float(cfg["training"].get("min_learning_rate", 1e-6)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg["training"]["epochs"])),
            eta_min=float(cfg["training"].get("min_learning_rate", 1e-6)),
        )
    robalrs_cfg = cfg.get("robalrs", {})
    loss_balancer = RobALRSLossBalancer(
        enabled=model_name == "pinn" and bool(robalrs_cfg.get("enabled", True)),
        ema=float(robalrs_cfg.get("ema", 0.98)),
        min_weight=float(robalrs_cfg.get("min_weight", 0.05)),
        max_weight=float(robalrs_cfg.get("max_weight", 20.0)),
        warmup_steps=int(robalrs_cfg.get("warmup_steps", 100)),
        min_weight_by_component=dict(robalrs_cfg.get("min_weight_by_component", {})),
        max_weight_by_component=dict(robalrs_cfg.get("max_weight_by_component", {})),
    )
    ema_model = None
    ema_decay = float(cfg["training"].get("ema_decay", 0.0) or 0.0)
    if ema_decay > 0.0:
        ema_model = ModelEMA(model, ema_decay)
        print(f"[STARTUP] model_ema_enabled decay={ema_decay}", flush=True)

    run_dir = resolve_checkpoint_dir(cfg, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] run_dir={run_dir}", flush=True)
    print("[STARTUP] initializing_wandb", flush=True)
    wb_run = init_wandb(model_name, cfg, run_dir)
    if wb_run is not None and bool(cfg.get("wandb", {}).get("log_code_artifacts", True)):
        print("[STARTUP] logging_wandb_artifacts", flush=True)
        log_run_artifacts(wb_run, model_name, run_dir, cfg, Path(cfg["data"]["stats_path"]))
    print("[STARTUP] entering_training_loop", flush=True)

    best_val = float("inf")
    best_epoch = 0
    best_val_result: dict[str, Any] = {}
    global_step = 0
    timing_log_every = int(cfg["training"].get("timing_log_every", 100))
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        totals: dict[str, float] = {"loss": 0.0}
        seen = 0
        timer = StepTimer()
        batch_ready_t = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"{model_name}/train", leave=False):
            step_t0 = time.perf_counter()
            timer.add("data_wait_sec", step_t0 - batch_ready_t)

            t0 = time.perf_counter()
            batch = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("to_device_sec", time.perf_counter() - t0)

            if model_name == "pinn":
                batch["element_pos"] = batch["element_pos"].detach().requires_grad_(True)

            t0 = time.perf_counter()
            pred = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("forward_sec", time.perf_counter() - t0)

            t0 = time.perf_counter()
            if model_name == "pinn":
                components = pinn_loss_components(pred, batch, cfg, stats, epoch=epoch, global_step=global_step + 1)
                loss, parts = loss_balancer.combine(components)
            else:
                loss, parts, _ = base_pressure_loss(pred, batch, cfg, stats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("loss_sec", time.perf_counter() - t0)

            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss.backward()
            clip = cfg["training"].get("grad_clip")
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("backward_sec", time.perf_counter() - t0)

            t0 = time.perf_counter()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("optimizer_sec", time.perf_counter() - t0)

            bsz = int(batch["pressure"].shape[0])
            seen += bsz
            global_step += 1
            totals["loss"] += float(loss.detach().cpu()) * bsz
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * bsz
            timer.add("step_total_sec", time.perf_counter() - step_t0)

            if global_step % timing_log_every == 0:
                timing_summary = timer.summary()
                if device.type == "cuda":
                    timing_summary["timing/cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                    timing_summary["timing/cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                    timing_summary["timing/cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                timing_summary["train/learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(
                    "[TIMING] "
                    + " ".join(f"{key}={value:.4f}" for key, value in timing_summary.items())
                )
                if wb_run is not None:
                    wb_run.log(timing_summary | {"global_step": global_step, "epoch": epoch})
                timer.reset()
            batch_ready_t = time.perf_counter()

        train_metrics = {key: value / max(1, seen) for key, value in totals.items()}
        do_val = epoch % int(cfg["training"].get("val_every", 1)) == 0
        if do_val:
            if ema_model is not None:
                with ema_model.apply_to(model):
                    val_result = evaluate_model(model, val_loader, device)
            else:
                val_result = evaluate_model(model, val_loader, device)
            val_loss = checkpoint_score(val_result, cfg)
            print_eval_summary(val_result, title=f"{model_name}/val epoch {epoch}")
        else:
            val_result = {}
            val_loss = train_metrics["loss"]
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6g} val_checkpoint_score={val_loss:.6g}")
        if wb_run is not None:
            wb_run.log(
                {f"train/{k}": v for k, v in train_metrics.items()}
                | flatten_metrics(val_result, prefix="val")
                | {"epoch": epoch, "train/learning_rate_epoch": float(optimizer.param_groups[0]["lr"])}
            )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "stats": stats,
            "train_metrics": train_metrics,
            "val_result": val_result,
            "checkpoint_score": val_loss,
        }
        torch.save(payload, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_val_result = val_result
            torch.save(payload, run_dir / "best.pt")
            if val_result:
                save_eval_report(val_result, run_dir / "best_eval.json")
                if wb_run is not None:
                    best_flat = flatten_metrics(val_result, prefix="best_val")
                    wb_run.summary["best_epoch"] = best_epoch
                    wb_run.summary["best_val_checkpoint_score"] = best_val
                    wb_run.summary["best_val_pressure_mse"] = val_result["pooled"]["pressure"]["mse"]
                    wb_run.summary.update(best_flat)
        patience = cfg["training"].get("early_stopping_patience")
        if patience and epoch - best_epoch >= int(patience):
            print(f"[EARLY STOP] no val improvement for {patience} epochs; best_epoch={best_epoch}")
            break

    if ema_model is not None:
        with ema_model.apply_to(model):
            final_result = evaluate_model(model, val_loader, device)
    else:
        final_result = evaluate_model(model, val_loader, device)
    save_eval_report(final_result, run_dir / "final_eval.json")
    if best_val_result:
        best_flat = flatten_metrics(best_val_result, prefix="best_val")
        best_report = {
            "best_epoch": best_epoch,
            "best_val_checkpoint_score": best_val,
            "best_val_pressure_mse": best_val_result["pooled"]["pressure"]["mse"],
            "best_val_pooled_pressure_r2": best_val_result["pooled"]["pressure"]["r2"],
            "best_val_unseen_pressure_r2": best_val_result.get("groups", {})
            .get("unseen_bases_10_11", {})
            .get("pressure", {})
            .get("r2", float("nan")),
            "best_val_unseen_center_of_pressure_r2": best_val_result.get("groups", {})
            .get("unseen_bases_10_11", {})
            .get("center_of_pressure", {})
            .get("r2", float("nan")),
            "best_val_unseen_peak_pressure_r2": best_val_result.get("groups", {})
            .get("unseen_bases_10_11", {})
            .get("peak_pressure", {})
            .get("r2", float("nan")),
            "best_val_unseen_reaction_proxy_r2": best_val_result.get("groups", {})
            .get("unseen_bases_10_11", {})
            .get("reaction_proxy", {})
            .get("r2", float("nan")),
        }
        (run_dir / "best_metrics.json").write_text(json.dumps(best_report | best_flat, indent=2, sort_keys=True))
        print(
            "[BEST_VAL] "
            + " ".join(
                [
                    f"epoch={best_epoch}",
                    f"score={best_val:.6g}",
                    f"pressure_mse={best_report['best_val_pressure_mse']:.6g}",
                    f"pooled_pressure_r2={best_report['best_val_pooled_pressure_r2']:.6g}",
                    f"unseen_pressure_r2={best_report['best_val_unseen_pressure_r2']:.6g}",
                    f"unseen_cop_r2={best_report['best_val_unseen_center_of_pressure_r2']:.6g}",
                ]
            ),
            flush=True,
        )
    else:
        best_flat = {}
        best_report = {"best_epoch": best_epoch, "best_val_pressure_mse": best_val}
    if wb_run is not None:
        wb_run.summary["best_val_checkpoint_score"] = best_val
        if best_val_result:
            wb_run.summary["best_val_pressure_mse"] = best_val_result["pooled"]["pressure"]["mse"]
        else:
            wb_run.summary["best_val_pressure_mse"] = best_val
        wb_run.summary["best_epoch"] = best_epoch
        if best_flat:
            wb_run.summary.update(best_flat)
            wb_run.log(
                best_flat
                | {
                    "best_val/epoch": best_epoch,
                    "best_val/selection_metric/checkpoint_score": best_val,
                    "best_val/selection_metric/pressure_mse": best_val_result["pooled"]["pressure"]["mse"],
                    "epoch": epoch,
                    "global_step": global_step,
                }
            )
        wb_run.summary.update(flatten_metrics(final_result, prefix="final_val"))
        if bool(cfg.get("wandb", {}).get("log_artifacts", True)):
            import wandb

            artifact = wandb.Artifact(f"{model_name}_contact_pressure_checkpoint", type="model")
            artifact.add_file(str(run_dir / "best.pt"))
            if (run_dir / "best_eval.json").exists():
                artifact.add_file(str(run_dir / "best_eval.json"))
            if (run_dir / "best_metrics.json").exists():
                artifact.add_file(str(run_dir / "best_metrics.json"))
            artifact.add_file(str(run_dir / "final_eval.json"))
            wb_run.log_artifact(artifact)
        wb_run.finish()
    train_ds.close()
    val_ds.close()
