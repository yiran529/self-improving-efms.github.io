from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    return nn.Sequential(*layers)


class TimerNet(nn.Module):
    """Two-head MLP: continuous action distribution + discrete distance distribution."""

    def __init__(
        self,
        obs_dim: int = 6,
        act_dim: int = 2,
        hidden_sizes: Sequence[int] = (256, 256, 256),
        num_distance_bins: int = 50,
        min_act_scale: float = 1e-2,
    ):
        super().__init__()
        self.min_act_scale = float(min_act_scale)
        self.act_backbone = build_mlp(obs_dim, hidden_sizes)
        self.dist_backbone = build_mlp(obs_dim, hidden_sizes)
        last_h = int(hidden_sizes[-1]) if len(hidden_sizes) > 0 else obs_dim

        self.act_loc = nn.Linear(last_h, act_dim)
        self.act_scale = nn.Linear(last_h, act_dim)
        self.dist_logits = nn.Linear(last_h, num_distance_bins, bias=False)

        # Matches notebook-style small init for action head.
        nn.init.normal_(self.act_loc.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.act_loc.bias, 0.0)
        nn.init.normal_(self.act_scale.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.act_scale.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_act = self.act_backbone(obs)
        h_dist = self.dist_backbone(obs)
        loc = self.act_loc(h_act)
        scale = F.softplus(self.act_scale(h_act)) + self.min_act_scale
        logits = self.dist_logits(h_dist)
        return loc, scale, logits

