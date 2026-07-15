from __future__ import annotations

from model_pinn import build_model as build_pinn
from train_common import run_training


def build_model(param_dim: int, cfg: dict, sample: dict):
    return build_pinn(param_dim, cfg, contact_dim=int(sample["contact_x"].shape[-1]))


if __name__ == "__main__":
    run_training("pinn", build_model)
