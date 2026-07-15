from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float = 0.0) -> nn.Sequential:
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, layers - 1)):
        blocks += [nn.Linear(dim, hidden_dim), nn.SiLU()]
        if dropout:
            blocks.append(nn.Dropout(dropout))
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


def von_mises_from_stress(stress: torch.Tensor) -> torch.Tensor:
    sxx, syy, szz, sxy, syz, sxz = stress.unbind(dim=-1)
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    vm2 = vm2 + 3.0 * (sxy.square() + syz.square() + sxz.square())
    return torch.sqrt(torch.clamp(vm2, min=1e-12)).unsqueeze(-1)


def build_knn_edges(points: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(points, points)
    knn = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]
    dst = torch.arange(points.shape[0], device=points.device).repeat_interleave(k)
    src = knn.reshape(-1)
    return torch.stack([src, dst], dim=0)


def pressure_projection(pressure: torch.Tensor, target_sum: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project predicted pressure so each sample has the target total pressure."""
    pressure = torch.relu(pressure)
    pred_sum = pressure.sum(dim=1, keepdim=True)
    scale = target_sum / torch.clamp(pred_sum, min=eps)
    return pressure * scale
