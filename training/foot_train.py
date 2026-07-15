from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from foot_data import PARAM_NAMES, FootShardDataset, compute_stats
from foot_models import FFNFootModel, GNOFootModel, PINNFootModel, von_mises_from_stress

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_model(model_name: str, cfg: dict[str, Any], train_ds: FootShardDataset) -> torch.nn.Module:
    mcfg = cfg["model"]
    common = {
        "param_dim": len(PARAM_NAMES),
        "hidden_dim": int(mcfg["hidden_dim"]),
        "layers": int(mcfg["layers"]),
        "dropout": float(mcfg.get("dropout", 0.0)),
    }
    if model_name == "ffn":
        return FFNFootModel(**common)
    if model_name == "pinn":
        return PINNFootModel(**common)
    if model_name == "gno":
        first = train_ds[0]
        return GNOFootModel(
            **common,
            reference_pos=first["element_pos"],
            k_neighbors=int(mcfg["k_neighbors"]),
        )
    raise ValueError(f"Unknown model {model_name!r}; expected gno, pinn, or ffn.")


def supervised_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    lcfg = cfg["loss"]
    elem_error = (pred["element_y"] - batch["element_y"]) ** 2
    elem_weight = 1.0 + batch["sole_mask"] * (float(lcfg["sole_element_weight"]) - 1.0)
    elem_loss = (elem_error * elem_weight).mean() * float(lcfg["element_weight"])

    contact_error = (pred["contact_y"] - batch["contact_y"]) ** 2
    contact_error[:, :, 1] *= float(lcfg["pressure_weight"])
    contact_loss = contact_error.mean() * float(lcfg["contact_weight"])

    total = elem_loss + contact_loss
    return total, {
        "element": float(elem_loss.detach().cpu()),
        "contact": float(contact_loss.detach().cpu()),
    }


def pinn_physics_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    lcfg = cfg["loss"]
    stress = pred["element_y"][:, :, :6]
    vm_pred = pred["element_y"][:, :, 6:7]
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

    # Stress order: sxx, syy, szz, sxy, syz, sxz.
    div_x = grads[0][:, :, 0] + grads[3][:, :, 1] + grads[5][:, :, 2]
    div_y = grads[3][:, :, 0] + grads[1][:, :, 1] + grads[4][:, :, 2]
    div_z = grads[5][:, :, 0] + grads[4][:, :, 1] + grads[2][:, :, 2]
    equilibrium = (div_x.square() + div_y.square() + div_z.square()).mean()

    vm_consistency = (vm_pred - von_mises_from_stress(stress)).square().mean()
    equilibrium = equilibrium * float(lcfg["pinn_equilibrium_weight"])
    vm_consistency = vm_consistency * float(lcfg["pinn_vm_consistency_weight"])
    return equilibrium + vm_consistency, {
        "equilibrium": float(equilibrium.detach().cpu()),
        "vm_consistency": float(vm_consistency.detach().cpu()),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {"loss": 0.0, "element": 0.0, "contact": 0.0}
    if cfg["training"]["model"] == "pinn" and is_train:
        totals.update({"equilibrium": 0.0, "vm_consistency": 0.0})

    count = 0
    iterator = tqdm(loader, leave=False, desc="train" if is_train else "val")
    for batch in iterator:
        batch = to_device(batch, device)
        if cfg["training"]["model"] == "pinn" and is_train:
            batch["element_pos"] = batch["element_pos"].detach().requires_grad_(True)

        with torch.set_grad_enabled(is_train):
            pred = model(batch)
            loss, parts = supervised_loss(pred, batch, cfg)
            if cfg["training"]["model"] == "pinn" and is_train:
                physics, physics_parts = pinn_physics_loss(pred, batch, cfg)
                loss = loss + physics
                parts.update(physics_parts)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = cfg["training"].get("grad_clip")
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()

        batch_size = int(batch["params"].shape[0])
        count += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FEBio foot surrogate models.")
    parser.add_argument("--config", default="training/foot_config.yaml")
    parser.add_argument("--model", choices=["gno", "pinn", "ffn"], default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["training"]["model"] = args.model
    if args.shard_dir:
        cfg["data"]["shard_dir"] = args.shard_dir
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.max_samples:
        cfg["data"]["train_max_samples"] = args.max_samples
        cfg["data"]["val_max_samples"] = max(1, args.max_samples // 5)

    stats_path = Path(cfg["data"]["stats_path"])
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = compute_stats(
            cfg["data"]["shard_dir"],
            out_path=stats_path,
            max_samples=cfg["data"].get("stats_max_samples", 2000),
        )

    train_ds = FootShardDataset(
        cfg["data"]["shard_dir"],
        split="train",
        stats=stats,
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["data"]["seed"]),
        max_samples=cfg["data"].get("train_max_samples"),
    )
    val_ds = FootShardDataset(
        cfg["data"]["shard_dir"],
        split="val",
        stats=stats,
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["data"]["seed"]),
        max_samples=cfg["data"].get("val_max_samples"),
    )

    device = choose_device(str(cfg["training"]["device"]))
    model = build_model(cfg["training"]["model"], cfg, train_ds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["model"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, cfg, device, None)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6g} "
            f"val_loss={val_metrics['loss']:.6g} "
            f"train_contact={train_metrics['contact']:.6g} "
            f"val_contact={val_metrics['contact']:.6g}"
        )
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "stats": stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(payload, ckpt_dir / "best.pt")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
