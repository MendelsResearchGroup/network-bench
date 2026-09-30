"""A per-bead feed-forward baseline: the same features, no message passing.

Identical node encoding to `GNS` -- the axis-shared MLP over the velocity history
and whatever extra channels the run asked for -- and then a decoder, with the
processor stack removed. Each bead is predicted from its own history alone, so
the graph is never consulted.

That makes it the honest control for the graph itself. The gap between `mlp` and
`gns` at the same hidden size is how much of the score comes from neighbours
rather than from the per-bead signal, which is not something the `frozen` or
`linear_floor` rows can tell you: those bound the trivial predictions, this one
bounds the non-trivial local one.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import SimulatorModel
from ..normalization import Normalizer
from .blocks import AxisSharedEncoder, build_mlp, node_extras

__all__ = ["NodeMLP"]


class NodeMLP(SimulatorModel):
    """Per-node feed-forward predictor of one-step acceleration."""

    def __init__(
        self,
        spec: InputGraphSpec,
        target_scale: Normalizer,
        *,
        hidden_size: int = 64,
        num_mlp: int = 3,
    ):
        super().__init__(spec, target_scale, hidden_size=hidden_size, num_mlp=num_mlp)

        self.node_encoder = AxisSharedEncoder(
            spec.history, hidden_size, dim=spec.dim, num_mlp=num_mlp, extra_channels=spec.extra_node_channels
        )
        self.node_projection = torch.nn.Linear(hidden_size * spec.dim, hidden_size)
        self.decoder = build_mlp(hidden_size, hidden_size, spec.dim, num_mlp=num_mlp)

        self.node_normalizer = Normalizer(spec.node_feature_width)
        self.force_normalizer = Normalizer(spec.dim) if spec.node_force else None

    def input_normalizers(self) -> list[Normalizer]:
        return [n for n in (self.node_normalizer, self.force_normalizer) if n is not None]

    def predict_acceleration(self, graph: Data) -> Tensor:
        extras = node_extras(graph, self.spec)
        if self.force_normalizer is not None:
            normalized = graph.clone()
            normalized.node_force = self.force_normalizer(graph.node_force, accumulate=self.training)
            extras = node_extras(normalized, self.spec)

        x = self.node_normalizer(graph.x, accumulate=self.training)
        x = self.node_projection(self.node_encoder(x, extras))
        return self.target_scale.inverse(self.decoder(x))
