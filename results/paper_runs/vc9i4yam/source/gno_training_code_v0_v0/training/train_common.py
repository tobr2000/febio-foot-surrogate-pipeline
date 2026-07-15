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

from foot_data import PARAM_NAMES, FootShardDataset, compute_stats
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
    with file_lock(lock_path):
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                if required.issubset(stats.keys()) and stats.get("param_names") == PARAM_NAMES:
                    return stats
            except json.JSONDecodeError:
                print(f"[WARN] Stats file {stats_path} is incomplete/corrupt; recomputing.")
        return compute_stats(
            cfg["data"]["shard_dir"],
            out_path=stats_path,
            max_samples=cfg["data"].get("stats_max_samples", 2000),
        )


def resolve_checkpoint_dir(cfg: dict[str, Any], model_name: str) -> Path:
    root = Path(cfg["training"]["checkpoint_dir"])
    suffix = os.environ.get("RUN_NAME_SUFFIX", "").strip()
    run_name = f"{model_name}_{suffix}" if suffix else model_name
    return root / run_name


def dataset_fingerprint(shard_dir: str | Path) -> dict[str, Any]:
    root = Path(shard_dir)
    paths = sorted(root.glob("batch_*.npz"))
    h = hashlib.sha256()
    total_size = 0
    total_samples = 0
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
                total_samples += int(sample_ids.size)
                if sample_ids.size:
                    local_min = int(sample_ids.min())
                    local_max = int(sample_ids.max())
                    sample_min = local_min if sample_min is None else min(sample_min, local_min)
                    sample_max = local_max if sample_max is None else max(sample_max, local_max)
                    h.update(f"{rel}|samples|{sample_ids.size}|{local_min}|{local_max}\n".encode("utf-8"))
        except Exception as exc:
            h.update(f"{rel}|sample_ids_unavailable|{type(exc).__name__}\n".encode("utf-8"))
    return {
        "shard_dir": str(root),
        "shard_count": len(paths),
        "packed_sample_count": total_samples,
        "sample_id_min": sample_min,
        "sample_id_max": sample_max,
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
    )
    device = choose_device(str(cfg["training"]["device"]))
    num_workers = int(tcfg["num_workers"])
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(tcfg.get("persistent_workers", False)) and num_workers > 0,
    }
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
    return {
        "pressure": pressure * float(lcfg["pressure_weight"]),
        "reaction": reaction * float(lcfg["reaction_weight"]),
        "peak": peak * float(lcfg["peak_weight"]),
        "nonnegative_pressure": nonnegative * float(lcfg.get("nonnegative_pressure_weight", 0.02)),
    }


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


def pinn_component_multiplier(name: str, cfg: dict[str, Any], epoch: int) -> float:
    pcfg = cfg.get("pinn_curriculum", {})
    if name in {"pressure", "reaction", "peak", "nonnegative_pressure"}:
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

    if "node_displacement" in pred:
        parts["node_displacement_aux"] = torch.mean((pred["node_displacement"] - batch["node_disp"]).square()) * float(
            lcfg["node_aux_weight"]
        )

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
    return {
        key: value * pinn_component_multiplier(key, cfg, epoch)
        for key, value in parts.items()
        if pinn_component_multiplier(key, cfg, epoch) > 0.0
    }


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
    ) -> None:
        self.enabled = enabled
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.step = 0
        self.loss_ema: dict[str, float] = {}
        self.weights: dict[str, float] = {}

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
            self.weights = {
                key: float(np.clip(raw[key] / max(mean_raw, self.eps), self.min_weight, self.max_weight))
                for key in components
            }
            mean_weight = sum(self.weights.values()) / max(1, len(self.weights))
            self.weights = {key: value / max(mean_weight, self.eps) for key, value in self.weights.items()}
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

    set_seed(int(cfg["data"]["seed"]))
    cfg["data"]["fingerprint"] = dataset_fingerprint(cfg["data"]["shard_dir"])
    stats = ensure_stats(cfg)
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, stats)
    device = choose_device(str(cfg["training"]["device"]))
    sample = train_ds[0]
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
    )

    run_dir = resolve_checkpoint_dir(cfg, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    wb_run = init_wandb(model_name, cfg, run_dir)
    if wb_run is not None and bool(cfg.get("wandb", {}).get("log_code_artifacts", True)):
        log_run_artifacts(wb_run, model_name, run_dir, cfg, Path(cfg["data"]["stats_path"]))

    best_val = float("inf")
    best_epoch = 0
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
                components = pinn_loss_components(pred, batch, cfg, stats, epoch=epoch)
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
            val_result = evaluate_model(model, val_loader, device)
            val_loss = val_result["pooled"]["pressure"]["mse"]
            print_eval_summary(val_result, title=f"{model_name}/val epoch {epoch}")
        else:
            val_result = {}
            val_loss = train_metrics["loss"]
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6g} val_pressure_mse={val_loss:.6g}")
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
        }
        torch.save(payload, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(payload, run_dir / "best.pt")
            if val_result:
                save_eval_report(val_result, run_dir / "best_eval.json")
                if wb_run is not None:
                    wb_run.summary["best_epoch"] = best_epoch
                    wb_run.summary["best_val_pressure_mse"] = best_val
                    wb_run.summary.update(flatten_metrics(val_result, prefix="best_val"))
        patience = cfg["training"].get("early_stopping_patience")
        if patience and epoch - best_epoch >= int(patience):
            print(f"[EARLY STOP] no val improvement for {patience} epochs; best_epoch={best_epoch}")
            break

    final_result = evaluate_model(model, val_loader, device)
    save_eval_report(final_result, run_dir / "final_eval.json")
    if wb_run is not None:
        wb_run.summary["best_val_pressure_mse"] = best_val
        wb_run.summary["best_epoch"] = best_epoch
        wb_run.summary.update(flatten_metrics(final_result, prefix="final_val"))
        if bool(cfg.get("wandb", {}).get("log_artifacts", True)):
            import wandb

            artifact = wandb.Artifact(f"{model_name}_contact_pressure_checkpoint", type="model")
            artifact.add_file(str(run_dir / "best.pt"))
            artifact.add_file(str(run_dir / "final_eval.json"))
            wb_run.log_artifact(artifact)
        wb_run.finish()
    train_ds.close()
    val_ds.close()
