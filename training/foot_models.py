from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float = 0.0) -> nn.Sequential:
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, layers - 1)):
        blocks += [nn.Linear(dim, hidden_dim), nn.GELU()]
        if dropout:
            blocks.append(nn.Dropout(dropout))
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


class FFNFootModel(nn.Module):
    def __init__(self, param_dim: int, hidden_dim: int = 128, layers: int = 4, dropout: float = 0.05):
        super().__init__()
        self.element_net = mlp(3 + 1 + param_dim, hidden_dim, 7, layers, dropout)
        self.contact_net = mlp(2 + param_dim, hidden_dim, 2, layers, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        pos = batch["element_pos"]
        mask = batch["sole_mask"]
        bsz, n_elem, _ = pos.shape
        elem_params = params[:, None, :].expand(bsz, n_elem, -1)
        elem_in = torch.cat([pos, mask, elem_params], dim=-1)

        contact_x = batch["contact_x"]
        n_contact = contact_x.shape[1]
        contact_params = params[:, None, :].expand(bsz, n_contact, -1)
        contact_in = torch.cat([contact_x, contact_params], dim=-1)
        return {
            "element_y": self.element_net(elem_in),
            "contact_y": self.contact_net(contact_in),
        }


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = mlp(hidden_dim * 2 + 3, hidden_dim, hidden_dim, 2, dropout)
        self.update = mlp(hidden_dim * 2, hidden_dim, hidden_dim, 2, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src]
        h_dst = h[:, dst]
        delta = pos[:, dst] - pos[:, src]
        msg = self.message(torch.cat([h_src, h_dst, delta], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, msg)
        out = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + out)


def build_knn_edges(points: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(points, points)
    knn = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]
    dst = torch.arange(points.shape[0], device=points.device).repeat_interleave(k)
    src = knn.reshape(-1)
    return torch.stack([src, dst], dim=0)


class GNOFootModel(nn.Module):
    def __init__(
        self,
        param_dim: int,
        reference_pos: torch.Tensor,
        hidden_dim: int = 128,
        layers: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("edge_index", build_knn_edges(reference_pos.float(), k_neighbors))
        self.embed = mlp(3 + 1 + param_dim, hidden_dim, hidden_dim, 2, dropout)
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, dropout) for _ in range(layers)])
        self.element_head = mlp(hidden_dim, hidden_dim, 7, 2, dropout)
        self.contact_net = mlp(2 + param_dim, hidden_dim, 2, layers, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        pos = batch["element_pos"]
        mask = batch["sole_mask"]
        bsz, n_elem, _ = pos.shape
        elem_params = params[:, None, :].expand(bsz, n_elem, -1)
        h = self.embed(torch.cat([pos, mask, elem_params], dim=-1))
        for block in self.blocks:
            h = block(h, pos, self.edge_index)

        contact_x = batch["contact_x"]
        n_contact = contact_x.shape[1]
        contact_params = params[:, None, :].expand(bsz, n_contact, -1)
        contact_in = torch.cat([contact_x, contact_params], dim=-1)
        return {
            "element_y": self.element_head(h),
            "contact_y": self.contact_net(contact_in),
        }


class PINNFootModel(nn.Module):
    def __init__(self, param_dim: int, hidden_dim: int = 128, layers: int = 5, dropout: float = 0.0):
        super().__init__()
        self.field_net = mlp(3 + 1 + param_dim, hidden_dim, 7, layers, dropout)
        self.contact_net = mlp(2 + param_dim, hidden_dim, 2, layers, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params = batch["params"]
        pos = batch["element_pos"]
        mask = batch["sole_mask"]
        bsz, n_elem, _ = pos.shape
        elem_params = params[:, None, :].expand(bsz, n_elem, -1)
        elem_in = torch.cat([pos, mask, elem_params], dim=-1)

        contact_x = batch["contact_x"]
        n_contact = contact_x.shape[1]
        contact_params = params[:, None, :].expand(bsz, n_contact, -1)
        contact_in = torch.cat([contact_x, contact_params], dim=-1)
        return {
            "element_y": self.field_net(elem_in),
            "contact_y": self.contact_net(contact_in),
        }


def von_mises_from_stress(stress: torch.Tensor) -> torch.Tensor:
    sxx, syy, szz, sxy, syz, sxz = stress.unbind(dim=-1)
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    vm2 = vm2 + 3.0 * (sxy**2 + syz**2 + sxz**2)
    return torch.sqrt(torch.clamp(vm2, min=1e-12)).unsqueeze(-1)
