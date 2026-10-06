"""The model interface: subclass `SimulatorModel` to be benchmarked.

    from gnn_physics_benchmark.interfaces import SimulatorModel

    class MyModel(SimulatorModel):
        def __init__(self, spec, target_scale, *, hidden_size: int = 64):
            super().__init__(spec, target_scale, hidden_size=hidden_size)
            self.net = torch.nn.Linear(spec.node_feature_width, spec.dim)

        def predict_acceleration(self, graph):
            return self.target_scale.inverse(self.net(graph.x))   # [N, dim]

A model is handed one input graph (see `interfaces.graph`) and predicts the
acceleration of every node. `forward` turns that into the next state of the
trajectory -- a frame in the raw dataset schema -- so a rollout can feed it
straight back in. The benchmark builds the input, owns the loss, runs the
rollout and computes the metrics.

`integrate_acceleration` stashes the acceleration on the frame it builds, and the
loss reads it back from there: positions are O(3) sigma while the acceleration is
O(1e-6), so recovering it by differencing float32 positions would lose digits.

The box is not predicted. Compression is imposed externally and the free axes
follow a barostat, so the next frame carries the input frame's box and the
rollout replaces it; see `gnn_physics_benchmark.barostat`.

A model whose whole idea is a different target normalisation may also define
`normalize_target(acceleration, *, accumulate) -> Tensor`; the loss is then taken
through it instead of the shared scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from ..normalization import Normalizer
    from .graph import InputGraphSpec

__all__ = ["SimulatorModel", "integrate_acceleration", "frame_from_positions", "predicted_acceleration", "current_velocity"]

#: Fields copied from the input graph onto the predicted frame unchanged.
_CARRIED = ("box", "box_tensor", "atom_ids", "atom_types", "molecule_ids")


class SimulatorModel(torch.nn.Module, ABC):
    """Base class of every benchmarked model.

    Subclasses implement `predict_acceleration`, and pass their hyperparameters
    to this constructor by keyword: they become `self.hyperparameters`, which
    names the model's choices in the results. Anything about the *input* --
    history length, which features, the graph radius -- belongs to the run's
    `InputGraphSpec` in `self.spec`, not here.

    `target_scale` is the acceleration normaliser the benchmark fitted for this
    run. A model decodes in standardised space and applies
    `self.target_scale.inverse` to return physical units, which is what makes
    one model's loss comparable with another's.
    """

    def __init__(self, spec: InputGraphSpec, target_scale: Normalizer, **hyperparameters: Any):
        super().__init__()
        self.spec = spec
        self.target_scale = target_scale
        self.hyperparameters = hyperparameters

    @abstractmethod
    def predict_acceleration(self, graph: Data) -> Tensor:
        """The acceleration of every node, `[N, dim]`, in physical units."""

    def forward(self, graph: Data) -> Data:
        """The next frame, in the raw dataset schema."""
        return integrate_acceleration(graph, self.predict_acceleration(graph))

    def input_normalizers(self) -> list[Normalizer]:
        """Normalisers the training loop freezes at `freeze_input_norm_epoch`."""
        return []


def current_velocity(graph: Data) -> Tensor:
    """`x_t - x_{t-1}` of the newest input frame, in sigma per frame."""
    return graph.pos - graph.prev_pos


def integrate_acceleration(graph: Data, acceleration: Tensor) -> Data:
    """Build the next frame from a predicted acceleration.

    Semi-implicit Euler with a stored frame as the unit and stride `s`:

        x_{t+s} = x_t + s * (x_t - x_{t-1}) + s**2 * a

    The result is a frame in the raw dataset schema -- bond-only directed edges
    with their 5-wide features recomputed at the new positions -- so it can be
    appended to a window and prepared exactly like a stored frame. The predicted
    acceleration is stashed on it as `acceleration` so the loss does not have to
    difference float32 positions to get it back.
    """
    if acceleration.shape != graph.pos.shape:
        raise ValueError(
            f"acceleration has shape {tuple(acceleration.shape)}, expected {tuple(graph.pos.shape)}."
        )

    stride = getattr(graph, "prediction_stride", 1)
    position = graph.pos + stride * current_velocity(graph) + stride**2 * acceleration
    frame = frame_from_positions(graph, position, stride)
    frame.acceleration = acceleration
    return frame


def frame_from_positions(graph: Data, position: Tensor, frame_offset: int, box: Tensor | None = None) -> Data:
    """Build a raw frame at given positions, carrying topology and identity.

    `box` is the new frame's box; by default it keeps the input graph's.
    """

    # Recompute the raw bond features at the new positions, under the minimum
    # image. Imported here to keep this module free of the graph machinery.
    from ..graph.build import minimum_image

    box = graph.box_tensor if box is None else box
    source, target = graph.bond_index
    vectors = minimum_image(position[source] - position[target], box)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)

    frame = Data(
        x=position,
        pos=position,
        edge_index=graph.bond_index,
        edge_attr=torch.cat([vectors, lengths, graph.bond_attr[:, position.shape[1] + 1 :]], dim=1),
        bond_types=graph.bond_types,
        time=int(graph.time) + frame_offset * int(graph.frame_interval),
    )
    for field in _CARRIED:
        value = getattr(graph, field, None)
        if value is not None:
            setattr(frame, field, value)
    frame.box_tensor = box
    return frame


def predicted_acceleration(frame: Data, graph: Data) -> Tensor:
    """The acceleration a predicted frame represents, relative to its input graph.

    Prefers the value `integrate_acceleration` stashed; falls back to
    differencing positions for a model that built its frame by hand, at the cost
    of the float32 digits described in this module's docstring.
    """
    stashed = getattr(frame, "acceleration", None)
    if stashed is not None:
        return stashed
    stride = getattr(graph, "prediction_stride", 1)
    return (frame.pos - graph.pos - stride * current_velocity(graph)) / stride**2
