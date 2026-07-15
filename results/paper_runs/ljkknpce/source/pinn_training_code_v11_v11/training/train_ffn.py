from __future__ import annotations

from model_ffn import build_model as build_ffn
from train_common import run_training


def build_model(param_dim: int, cfg: dict, sample: dict):
    return build_ffn(param_dim, cfg, contact_dim=int(sample["contact_x"].shape[-1]))


if __name__ == "__main__":
    run_training("ffn", build_model)
