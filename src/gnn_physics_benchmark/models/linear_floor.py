"""The best linear predictor of acceleration from the velocity history.

This is the dataset's difficulty floor turned into a registered model, so that it
appears as a row in the same results table as everything else. It predicts

    a = sum_k w_k * v_{t-k}

with one scalar weight per history step, shared across the three spatial axes --
exactly the family that `metrics.ladder.linear_ladder` fits in closed form. Fitted
by gradient descent here rather than by least squares, so it goes through the same
training loop as every other model and its number is comparable for the same
reasons.

A GNN that does not beat this is not learning the material's response; it is
reproducing the trivial persistence of the velocity. Because the input features
are already on the target's own scale -- residual velocities are O(1e-5) against
an acceleration of O(3e-6) -- the weights are O(0.1) and no normalisation is needed.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import integrate_acceleration
from ..normalization import Normalizer

__all__ = ["LinearFloor"]


class LinearFloor(torch.nn.Module):
    """One weight per history step, shared across axes."""

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer, *, use_extras: bool = False):
        super().__init__()
        self.spec = spec
        self.use_extras = use_extras and spec.extra_node_channels > 0
        self.hyperparameters = {"use_extras": self.use_extras}
        channels = spec.history + (spec.extra_node_channels if self.use_extras else 0)
        self.weights = torch.nn.Parameter(torch.zeros(channels))

    def input_normalizers(self) -> list[Normalizer]:
        return []

    def forward(self, graph: Data) -> Data:
        # [N, history * dim] -> [N, dim, history], matching the axis-shared layout.
        per_axis = graph.x.view(graph.x.size(0), self.spec.history, self.spec.dim).transpose(1, 2)
        if self.use_extras:
            from .blocks import node_extras

            per_axis = torch.cat([per_axis, node_extras(graph, self.spec)], dim=2)
        return integrate_acceleration(graph, per_axis @ self.weights)
