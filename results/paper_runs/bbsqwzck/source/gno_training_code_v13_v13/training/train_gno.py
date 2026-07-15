from __future__ import annotations

from model_gno import build_model as build_gno
from train_common import run_training


def build_model(param_dim: int, cfg: dict, sample: dict):
    return build_gno(
        param_dim,
        cfg,
        reference_contact_x=sample["contact_x"],
        reference_contact_geom_x=sample.get("contact_geom_x", sample["contact_x"]),
    )


if __name__ == "__main__":
    run_training("gno", build_model)
