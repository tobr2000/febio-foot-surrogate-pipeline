from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models_common import build_knn_edges, mlp


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, delta_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = mlp(hidden_dim * 2 + delta_dim, hidden_dim, hidden_dim, 2, dropout)
        self.update = mlp(hidden_dim * 2, hidden_dim, hidden_dim, 2, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src]
        h_dst = h[:, dst]
        delta = edge_x[:, dst] - edge_x[:, src]
        msg = self.message(torch.cat([h_src, h_dst, delta], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, msg)
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


class GNOContactPressureModel(nn.Module):
    """Graph neural operator style contact-pressure predictor over the sole contact surface.

    The model has two coupled branches:
    - a graph branch that predicts local spatial residuals over the contact mesh;
    - a global/region branch that predicts low-dimensional regional baselines and
      calibration scalars. This gives the graph model the same kind of global
      leverage the FFN baseline has shown on this dataset, while preserving the
      geometric contact-surface operator.
    """

    def __init__(
        self,
        param_dim: int,
        reference_contact_x: torch.Tensor,
        reference_contact_geom_x: torch.Tensor | None = None,
        hidden_dim: int = 192,
        layers: int = 5,
        k_neighbors: int = 10,
        dropout: float = 0.05,
        input_noise_std: float = 0.0,
        param_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_noise_std = float(input_noise_std)
        self.param_dropout = nn.Dropout(float(param_dropout)) if param_dropout else nn.Identity()
        if reference_contact_geom_x is None:
            reference_contact_geom_x = reference_contact_x
        self.contact_dim = int(reference_contact_x.shape[-1])
        self.geom_dim = int(reference_contact_geom_x.shape[-1])
        self.region_dim = 4 if self.contact_dim - self.geom_dim >= 4 else 0
        self.register_buffer("edge_index", build_knn_edges(reference_contact_geom_x.float(), k_neighbors))
        self.embed = mlp(self.contact_dim + param_dim, hidden_dim, hidden_dim, 2, dropout)
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, self.geom_dim, dropout) for _ in range(layers)])
        self.shape_head = mlp(hidden_dim, hidden_dim, 1, 2, dropout)
        self.residual_gate = mlp(hidden_dim, hidden_dim, 1, 2, dropout)
        self.graph_head = mlp(hidden_dim * 2 + param_dim, hidden_dim, 6, 3, dropout)
        self.region_head = mlp(hidden_dim * 2 + param_dim, hidden_dim, max(1, self.region_dim), 3, dropout)

    def _region_features(self, contact_x: torch.Tensor) -> torch.Tensor | None:
        if self.region_dim <= 0 or contact_x.shape[-1] < self.geom_dim + self.region_dim:
            return None
        return contact_x[..., self.geom_dim : self.geom_dim + self.region_dim]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        contact_x = batch["contact_x"]
        edge_x = batch.get("contact_geom_x", contact_x)
        bsz, n_contact, _ = contact_x.shape
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(contact_x) * self.input_noise_std
            if contact_x.shape[-1] > edge_x.shape[-1]:
                noise[..., edge_x.shape[-1]:] = 0.0
            contact_x = contact_x + noise
        params = self.param_dropout(params)
        param_features = params[:, None, :].expand(bsz, n_contact, -1)
        h = self.embed(torch.cat([contact_x, param_features], dim=-1))
        for block in self.blocks:
            h = block(h, edge_x, self.edge_index)

        pressure_shape = self.shape_head(h)
        residual_gate = torch.sigmoid(self.residual_gate(h))
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1), params], dim=-1)
        graph_out = self.graph_head(pooled)
        pressure_scale = 0.35 + F.softplus(graph_out[:, 0:1])
        pressure_shift = 0.1 * torch.tanh(graph_out[:, 1:2])
        reaction_mean_norm = graph_out[:, 2:3]
        peak_norm = graph_out[:, 3:4]
        residual_scale = 0.1 + F.softplus(graph_out[:, 4:5])
        baseline_mix = torch.sigmoid(graph_out[:, 5:6])

        region = self._region_features(contact_x)
        region_logits = self.region_head(pooled)
        if region is not None:
            region_baseline = torch.sum(region * region_logits[:, None, : self.region_dim], dim=-1, keepdim=True)
        else:
            region_baseline = torch.zeros_like(pressure_shape)

        pressure_residual = residual_gate * pressure_shape
        pressure = (
            baseline_mix[:, None, :] * region_baseline
            + (1.0 - baseline_mix[:, None, :]) * pressure_shift[:, None, :]
            + residual_scale[:, None, :] * pressure_residual
        )
        pressure = pressure * pressure_scale[:, None, :]
        return {
            "pressure": pressure,
            "pressure_shape": pressure_shape,
            "pressure_residual": pressure_residual,
            "region_baseline": region_baseline,
            "region_logits": region_logits[:, : self.region_dim] if self.region_dim > 0 else region_logits[:, :0],
            "pressure_scale": pressure_scale,
            "pressure_shift": pressure_shift,
            "residual_scale": residual_scale,
            "baseline_mix": baseline_mix,
            "reaction_mean_norm": reaction_mean_norm,
            "peak_norm": peak_norm,
        }


def build_model(param_dim: int, cfg: dict, reference_contact_x: torch.Tensor, reference_contact_geom_x: torch.Tensor | None = None) -> GNOContactPressureModel:
    mcfg = cfg["model"]
    return GNOContactPressureModel(
        param_dim=param_dim,
        reference_contact_x=reference_contact_x,
        reference_contact_geom_x=reference_contact_geom_x,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        k_neighbors=int(mcfg["k_neighbors"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        input_noise_std=float(mcfg.get("input_noise_std", 0.0)),
        param_dropout=float(mcfg.get("param_dropout", 0.0)),
    )
