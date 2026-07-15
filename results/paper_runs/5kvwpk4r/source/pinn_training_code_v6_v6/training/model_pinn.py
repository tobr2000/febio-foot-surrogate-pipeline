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

    def __init__(self, param_dim: int, contact_dim: int = 2, hidden_dim: int = 128, layers: int = 5, dropout: float = 0.0):
        super().__init__()
        self.contact_dim = int(contact_dim)
        self.field_net = mlp(4 + param_dim, hidden_dim, 10, layers, dropout)
        self.contact_net = mlp(self.contact_dim + 1 + param_dim + 3, hidden_dim, 1, layers, dropout)
        self.global_encoder = mlp(param_dim, hidden_dim, 3, 3, dropout)

    def field_at(self, coords: torch.Tensor, params: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, n_points, _ = coords.shape
        point_params = params[:, None, :].expand(bsz, n_points, -1)
        point_times = times[:, None, None].expand(bsz, n_points, 1)
        fields = self.field_net(torch.cat([coords, point_times, point_params], dim=-1))
        stress = fields[:, :, 3:9]
        return {
            "displacement": fields[:, :, :3],
            "stress": stress,
            "von_mises": fields[:, :, 9:10],
            "von_mises_from_stress": von_mises_from_stress(stress),
        }

    def contact_at(
        self,
        contact_x: torch.Tensor,
        params: torch.Tensor,
        global_features: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_contact, _ = contact_x.shape
        contact_params = params[:, None, :].expand(bsz, n_contact, -1)
        contact_global = global_features[:, None, :].expand(bsz, n_contact, -1)
        contact_times = times[:, None, None].expand(bsz, n_contact, 1)
        return self.contact_net(torch.cat([contact_x, contact_times, contact_params, contact_global], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor], project_contact: bool = False) -> dict[str, torch.Tensor]:
        params = batch["params"]
        elem_pos = batch["element_pos"]
        bsz, n_elem, _ = elem_pos.shape
        final_time = torch.ones(bsz, device=params.device, dtype=params.dtype)
        elem_fields = self.field_at(elem_pos, params, final_time)

        node_pos = batch["node_pos"]
        n_nodes = node_pos.shape[1]
        node_fields = self.field_at(node_pos, params, final_time)

        global_features = self.global_encoder(params)
        contact_x = batch["contact_x"]
        pressure = self.contact_at(contact_x, params, global_features, final_time)

        if project_contact:
            target_sum = batch["pressure"].sum(dim=1, keepdim=True)
            pressure = pressure_projection(pressure, target_sum)

        out = {
            "pressure": pressure,
            "displacement": elem_fields["displacement"],
            "node_displacement": node_fields["displacement"],
            "stress": elem_fields["stress"],
            "von_mises": elem_fields["von_mises"],
            "von_mises_from_stress": elem_fields["von_mises_from_stress"],
        }
        if "element_history_pos" in batch:
            hist_pos = batch["element_history_pos"]
            hist_times = batch["element_history_times"].to(params.device, params.dtype)
            b_hist, t_hist, n_hist, _ = hist_pos.shape
            flat_pos = hist_pos.reshape(b_hist * t_hist, n_hist, 3)
            flat_params = params[:, None, :].expand(b_hist, t_hist, -1).reshape(b_hist * t_hist, -1)
            flat_times = hist_times.reshape(b_hist * t_hist)
            hist_fields = self.field_at(flat_pos, flat_params, flat_times)
            out["history_stress"] = hist_fields["stress"].reshape(b_hist, t_hist, n_hist, 6)
            out["history_von_mises"] = hist_fields["von_mises"].reshape(b_hist, t_hist, n_hist, 1)
            out["history_von_mises_from_stress"] = hist_fields["von_mises_from_stress"].reshape(b_hist, t_hist, n_hist, 1)
        if "node_history_pos" in batch:
            hist_pos = batch["node_history_pos"]
            hist_times = batch["node_history_times"].to(params.device, params.dtype)
            b_hist, t_hist, n_hist, _ = hist_pos.shape
            flat_pos = hist_pos.reshape(b_hist * t_hist, n_hist, 3)
            flat_params = params[:, None, :].expand(b_hist, t_hist, -1).reshape(b_hist * t_hist, -1)
            flat_times = hist_times.reshape(b_hist * t_hist)
            hist_fields = self.field_at(flat_pos, flat_params, flat_times)
            out["history_node_displacement"] = hist_fields["displacement"].reshape(b_hist, t_hist, n_hist, 3)
        if "contact_history_y" in batch:
            hist_times = batch["contact_history_times"].to(params.device, params.dtype)
            hist_contact_x = batch.get("contact_history_x", contact_x)
            b_hist, t_hist = hist_times.shape
            flat_contact = hist_contact_x[:, None, :, :].expand(b_hist, t_hist, -1, -1).reshape(b_hist * t_hist, -1, hist_contact_x.shape[-1])
            flat_params = params[:, None, :].expand(b_hist, t_hist, -1).reshape(b_hist * t_hist, -1)
            flat_global = global_features[:, None, :].expand(b_hist, t_hist, -1).reshape(b_hist * t_hist, -1)
            out["history_pressure"] = self.contact_at(
                flat_contact,
                flat_params,
                flat_global,
                hist_times.reshape(b_hist * t_hist),
            ).reshape(b_hist, t_hist, -1, 1)
        return out


def build_model(param_dim: int, cfg: dict, contact_dim: int = 2) -> PINNContactPressureModel:
    mcfg = cfg["model"]
    return PINNContactPressureModel(
        param_dim=param_dim,
        contact_dim=contact_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
