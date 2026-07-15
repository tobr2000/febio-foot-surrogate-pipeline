from __future__ import annotations

import torch
from torch import nn

from models_common import mlp


class FFNContactPressureModel(nn.Module):
    """Pointwise baseline: parameters + contact-face features -> pressure."""

    def __init__(self, param_dim: int, contact_dim: int = 2, hidden_dim: int = 128, layers: int = 4, dropout: float = 0.05):
        super().__init__()
        self.net = mlp(contact_dim + param_dim, hidden_dim, 1, layers, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        contact_x = batch["contact_x"]
        bsz, n_contact, _ = contact_x.shape
        param_features = params[:, None, :].expand(bsz, n_contact, -1)
        pressure = self.net(torch.cat([contact_x, param_features], dim=-1))
        return {"pressure": pressure}


def build_model(param_dim: int, cfg: dict, contact_dim: int = 2) -> FFNContactPressureModel:
    mcfg = cfg["model"]
    return FFNContactPressureModel(
        param_dim=param_dim,
        contact_dim=contact_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
