"""A deliberately tiny per-node MLP from the velocity history to the acceleration.

One hidden layer of `hidden_dim` units over the flattened history, with the output
layer zero-initialised, so an untrained model predicts zero acceleration -- i.e.
starts out as pure velocity persistence and learns corrections to it.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import SimulatorModel
from ..normalization import Normalizer

__all__ = ["TinyVelocityMLP"]


class TinyVelocityMLP(SimulatorModel):
    """Map the velocity history `[N, dim * history]` to an acceleration `[N, dim]`."""

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer, *, hidden_dim: int = 4):
        super().__init__(spec, target_scale, hidden_dim=hidden_dim)

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

    def predict_acceleration(self, graph: Data) -> Tensor:
        x = self.node_normalizer(graph.x, accumulate=self.training)
        return self.target_scale.inverse(self.network(x))
