"""A per-node MLP that also reads the node's own bonds: one hop, no message passing.

Each bond's features (`dx, dy, r, stiffness, rest_length` on the spring networks)
go through a small edge MLP, and every node sums the encodings of its bonds. That
sum, next to the node's velocity history, feeds a node MLP that predicts the
acceleration. The node sees how its springs are stretched, but never its
neighbours' velocities and never anything two bonds away.

It sits between `tiny_mlp` (the node alone) and `gns` (ten rounds of message
passing). As in `tiny_mlp`, the output layer is zero-initialised, so an untrained
model predicts zero acceleration -- pure velocity persistence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import SimulatorModel
from ..normalization import Normalizer

__all__ = ["EdgeMLP"]


class EdgeMLP(SimulatorModel):
    """Velocity history plus summed bond encodings `->` acceleration `[N, dim]`."""

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer, *, hidden_dim: int = 128, depth: int = 4):
        super().__init__(spec, target_scale, hidden_dim=hidden_dim, depth=depth)

        # `depth` hidden layers in each of the two MLPs.
        edge_layers = []
        for index in range(depth):
            edge_layers += [nn.Linear(spec.edge_width if index == 0 else hidden_dim, hidden_dim), nn.GELU()]
        self.edge_network = nn.Sequential(*edge_layers, nn.Linear(hidden_dim, hidden_dim))

        node_layers = []
        for index in range(depth):
            width = spec.node_feature_width + hidden_dim if index == 0 else hidden_dim
            node_layers += [nn.Linear(width, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        self.node_network = nn.Sequential(*node_layers, nn.Linear(hidden_dim, spec.dim))
        nn.init.zeros_(self.node_network[-1].weight)
        nn.init.zeros_(self.node_network[-1].bias)

        self.node_normalizer = Normalizer(spec.node_feature_width)
        self.edge_normalizer = Normalizer(spec.edge_width)

    def input_normalizers(self) -> list[Normalizer]:
        return [self.node_normalizer, self.edge_normalizer]

    def predict_acceleration(self, graph: Data) -> Tensor:
        x = self.node_normalizer(graph.x, accumulate=self.training)
        edges = self.edge_network(self.edge_normalizer(graph.edge_attr, accumulate=self.training))
        # Sum each bond's encoding onto the node it points to.
        bonds = torch.zeros(x.size(0), edges.size(1), dtype=edges.dtype, device=edges.device)
        bonds.index_add_(0, graph.edge_index[1], edges)
        return self.target_scale.inverse(self.node_network(torch.cat([x, bonds], dim=1)))
