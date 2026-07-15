from __future__ import annotations

import torch
from torch import nn

from models_common import mlp, pressure_projection, von_mises_from_stress


class PINNContactPressureModel(nn.Module):
    """
    Hybrid PINN surrogate.

    The primary target is contact pressure. The auxiliary field network predicts
    displacement, stress, and von Mises inside the domain so we can penalize
    equilibrium and stress/von-Mises consistency.
    """

    def __init__(self, param_dim: int, hidden_dim: int = 128, layers: int = 5, dropout: float = 0.0):
        super().__init__()
        self.field_net = mlp(3 + param_dim, hidden_dim, 10, layers, dropout)
        self.contact_net = mlp(2 + param_dim + 3, hidden_dim, 1, layers, dropout)
        self.global_encoder = mlp(param_dim, hidden_dim, 3, 3, dropout)

    def forward(self, batch: dict[str, torch.Tensor], project_contact: bool = False) -> dict[str, torch.Tensor]:
        params = batch["params"]
        elem_pos = batch["element_pos"]
        bsz, n_elem, _ = elem_pos.shape
        elem_params = params[:, None, :].expand(bsz, n_elem, -1)
        fields = self.field_net(torch.cat([elem_pos, elem_params], dim=-1))
        displacement = fields[:, :, :3]
        stress = fields[:, :, 3:9]
        von_mises = fields[:, :, 9:10]

        node_pos = batch["node_pos"]
        n_nodes = node_pos.shape[1]
        node_params = params[:, None, :].expand(bsz, n_nodes, -1)
        node_fields = self.field_net(torch.cat([node_pos, node_params], dim=-1))
        node_displacement = node_fields[:, :, :3]

        global_features = self.global_encoder(params)
        contact_x = batch["contact_x"]
        n_contact = contact_x.shape[1]
        contact_params = params[:, None, :].expand(bsz, n_contact, -1)
        contact_global = global_features[:, None, :].expand(bsz, n_contact, -1)
        pressure = self.contact_net(torch.cat([contact_x, contact_params, contact_global], dim=-1))

        if project_contact:
            target_sum = batch["pressure"].sum(dim=1, keepdim=True)
            pressure = pressure_projection(pressure, target_sum)

        return {
            "pressure": pressure,
            "displacement": displacement,
            "node_displacement": node_displacement,
            "stress": stress,
            "von_mises": von_mises,
            "von_mises_from_stress": von_mises_from_stress(stress),
        }


def build_model(param_dim: int, cfg: dict) -> PINNContactPressureModel:
    mcfg = cfg["model"]
    return PINNContactPressureModel(
        param_dim=param_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
