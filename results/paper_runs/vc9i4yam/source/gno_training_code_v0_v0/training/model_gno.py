from __future__ import annotations

import torch
from torch import nn

from models_common import build_knn_edges, mlp


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = mlp(hidden_dim * 2 + 2, hidden_dim, hidden_dim, 2, dropout)
        self.update = mlp(hidden_dim * 2, hidden_dim, hidden_dim, 2, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, contact_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src]
        h_dst = h[:, dst]
        delta = contact_x[:, dst] - contact_x[:, src]
        msg = self.message(torch.cat([h_src, h_dst, delta], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, msg)
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


class GNOContactPressureModel(nn.Module):
    """Graph neural operator style contact-pressure predictor over the sole contact grid."""

    def __init__(
        self,
        param_dim: int,
        reference_contact_x: torch.Tensor,
        hidden_dim: int = 128,
        layers: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("edge_index", build_knn_edges(reference_contact_x.float(), k_neighbors))
        self.embed = mlp(2 + param_dim, hidden_dim, hidden_dim, 2, dropout)
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, dropout) for _ in range(layers)])
        self.head = mlp(hidden_dim, hidden_dim, 1, 2, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        contact_x = batch["contact_x"]
        bsz, n_contact, _ = contact_x.shape
        param_features = params[:, None, :].expand(bsz, n_contact, -1)
        h = self.embed(torch.cat([contact_x, param_features], dim=-1))
        for block in self.blocks:
            h = block(h, contact_x, self.edge_index)
        return {"pressure": self.head(h)}


def build_model(param_dim: int, cfg: dict, reference_contact_x: torch.Tensor) -> GNOContactPressureModel:
    mcfg = cfg["model"]
    return GNOContactPressureModel(
        param_dim=param_dim,
        reference_contact_x=reference_contact_x,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        k_neighbors=int(mcfg["k_neighbors"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
