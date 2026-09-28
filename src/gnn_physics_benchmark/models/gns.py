"""The reference encode-process-decode graph network simulator.

Ported from `KG_chains/simulator_model.py`. The architecture is the usual GNS
shape -- encode nodes and edges into a latent space, run `n_layers` rounds of
message passing with residual connections, decode a per-node vector -- with two
details that are specific to this problem:

* The node encoder is **axis-shared**: one MLP runs on every spatial axis, with
  the history steps as its input channels. See `blocks.AxisSharedEncoder`.
* `edge_attr` is **never updated** by message passing. It stays what the encoder
  produced, all the way through, so the processed information lives entirely on
  the nodes.

Input normalisation belongs to the model and lives here: node velocities, edge
features and the analytic force each get their own `Normalizer`, the force
separately because it is in epsilon/sigma while velocities are sigma per frame.
Fractional coordinates are passed through unnormalised, being already O(0.3).

The *output* scale is not the model's. The benchmark fits the acceleration
normaliser once, up front, and hands it in, so that every model's decoder works
in the same standardised space and their losses are comparable.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import integrate_acceleration
from ..normalization import Normalizer
from .blocks import AxisSharedEncoder, build_mlp, node_extras

__all__ = ["GNS"]


class _Processor(MessagePassing):
    """One round of message passing: build a message per edge, aggregate, update."""

    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__(aggr="add")
        self.edge_layer = build_mlp(hidden_size * 3, hidden_size, hidden_size * 3, num_mlp=num_mlp, layer_norm=True)
        self.node_layer = build_mlp(hidden_size * 4, hidden_size, hidden_size, num_mlp=num_mlp, layer_norm=True)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.edge_layer(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggregated: Tensor, x: Tensor) -> Tensor:
        return self.node_layer(torch.cat([aggregated, x], dim=1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)


class GNS(torch.nn.Module):
    """Graph network simulator predicting one-step acceleration."""

    def __init__(
        self,
        spec: InputGraphSpec,
        target_scale: Normalizer,
        *,
        hidden_size: int = 64,
        n_layers: int = 10,
        num_mlp: int = 3,
    ):
        super().__init__()
        self.spec = spec
        self.target_scale = target_scale
        self.hyperparameters = {"hidden_size": hidden_size, "n_layers": n_layers, "num_mlp": num_mlp}

        edge_width = spec.edge_width
        self.node_encoder = AxisSharedEncoder(
            spec.history, hidden_size, dim=spec.dim, num_mlp=num_mlp, extra_channels=spec.extra_node_channels
        )
        self.node_projection = torch.nn.Linear(hidden_size * spec.dim, hidden_size)
        self.edge_encoder = build_mlp(edge_width, hidden_size, hidden_size, num_mlp=num_mlp)
        self.processors = torch.nn.ModuleList(_Processor(hidden_size, num_mlp) for _ in range(n_layers))
        self.decoder = build_mlp(hidden_size, hidden_size, spec.dim, num_mlp=num_mlp)

        self.node_normalizer = Normalizer(spec.node_feature_width)
        self.edge_normalizer = Normalizer(edge_width)
        self.force_normalizer = Normalizer(spec.dim) if spec.node_force else None

    def input_normalizers(self) -> list[Normalizer]:
        """The normalisers the training loop freezes part-way through."""
        return [n for n in (self.node_normalizer, self.edge_normalizer, self.force_normalizer) if n is not None]

    def _normalized_extras(self, graph: Data) -> Tensor | None:
        if self.force_normalizer is None:
            return node_extras(graph, self.spec)
        normalized = graph.clone()
        normalized.node_force = self.force_normalizer(graph.node_force, accumulate=self.training)
        return node_extras(normalized, self.spec)

    def forward(self, graph: Data) -> Data:
        x = self.node_normalizer(graph.x, accumulate=self.training)
        edge_attr = self.edge_normalizer(graph.edge_attr, accumulate=self.training)

        x = self.node_projection(self.node_encoder(x, self._normalized_extras(graph)))
        edge_attr = self.edge_encoder(edge_attr)

        for processor in self.processors:
            x = x + processor(x, graph.edge_index, edge_attr)

        return integrate_acceleration(graph, self.target_scale.inverse(self.decoder(x)))
