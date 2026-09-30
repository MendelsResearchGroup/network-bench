"""A deliberately tiny per-node MLP from the velocity history to the acceleration.

One hidden layer of `hidden_dim` units over the flattened history, with the output
layer zero-initialised, so an untrained model predicts zero acceleration -- i.e.
starts out as pure velocity persistence and learns corrections to it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import integrate_acceleration
from ..normalization import Normalizer

__all__ = ["TinyVelocityMLP"]


class TinyVelocityMLP(nn.Module):
    """Map the velocity history `[N, dim * history]` to an acceleration `[N, dim]`."""

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer, *, hidden_dim: int = 4):
        super().__init__()
        self.spec = spec
        self.target_scale = target_scale
        self.hyperparameters = {"hidden_dim": hidden_dim}

        self.network = nn.Sequential(
            nn.Linear(spec.node_feature_width, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spec.dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

        # Velocities are O(1e-6) per frame; standardise them before the first layer.
        self.node_normalizer = Normalizer(spec.node_feature_width)

    def input_normalizers(self) -> list[Normalizer]:
        return [self.node_normalizer]

    def forward(self, graph: Data) -> Data:
        x = self.node_normalizer(graph.x, accumulate=self.training)
        return integrate_acceleration(graph, self.target_scale.inverse(self.network(x)))
