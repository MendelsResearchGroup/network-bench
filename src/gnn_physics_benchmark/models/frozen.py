"""The do-nothing baseline: every bead stays exactly where it is.

Beware of two different things that both get called "frozen" in this literature.

* **Frozen positions** -- this model. The next frame equals the current one, so
  the predicted acceleration has to cancel the current velocity,
  `a = -(x_t - x_{t-1})`. This is the exact denominator of the `relative_mse`
  metric, which is why it is worth having as a registered model: a `frozen` run
  must score `relative_mse == 1.0` to the last digit, and if it does not then the
  harness is wrong rather than the model. It is also the cheapest possible
  end-to-end test of the whole pipeline.

* **Zero acceleration** -- constant velocity, which is a genuinely decent
  predictor here and is *not* this. That one is the "frozen" rung of the linear
  difficulty ladder, where the quantity being predicted is the acceleration
  rather than the position. `linear_floor` with its weights at zero is exactly
  it, which is also where it starts training.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import SimulatorModel, current_velocity
from ..normalization import Normalizer

__all__ = ["Frozen", "ZeroAcceleration"]


class Frozen(SimulatorModel):
    """Predicts that nothing moves."""

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer):
        super().__init__(spec, target_scale)
        # The training loop builds an optimizer over the model's parameters; one
        # unused scalar keeps that path honest without changing the prediction.
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def predict_acceleration(self, graph: Data) -> Tensor:
        return -current_velocity(graph) / getattr(graph, "prediction_stride", 1) + 0.0 * self.unused


class ZeroAcceleration(SimulatorModel):
    """Carry observed seed velocity, without a neural acceleration proposal."""

    def predict_acceleration(self, graph: Data) -> Tensor:
        return torch.zeros_like(graph.pos)
