"""The contract a model satisfies to be benchmarked.

A model is handed one input graph (see `interfaces.graph`) and returns the next
state of the trajectory, as a frame in the raw dataset schema. Nothing else is
required of it: the benchmark builds the input, owns the loss, runs the rollout
and computes the metrics.

    class MyModel(torch.nn.Module):
        hyperparameters = {"hidden_size": 64}

        def forward(self, graph):
            acceleration = self.net(graph.x)          # [N, 3]
            return integrate_acceleration(graph, acceleration)

Why the output is a *state* and not a tensor
--------------------------------------------
Returning the next frame makes the model self-describing: a rollout can feed the
output straight back in as the newest frame of the next window, and nothing
outside the model needs to know what parameterisation it predicted in. The cost
is that the acceleration a loss wants has to be recovered from the positions --
and positions are O(3) sigma while the acceleration is O(3e-5), so differencing
them in float32 throws away about two decimal digits. `integrate_acceleration`
therefore stashes the acceleration it was given on the frame it builds, and
`predicted_acceleration` prefers the stash, falling back to differencing for a
model that assembles its output by hand. A model that uses the helper loses
nothing to rounding.

The box is not predicted. Compression is imposed externally, so the next frame
carries the input frame's box and the rollout advances it.

The optional loss hook
----------------------
By default the benchmark z-scores the acceleration target with statistics it
accumulates itself, which is what makes one model's loss comparable with
another's. A model whose whole idea is a different target normalisation -- a
per-system rescaling, say -- may define

    def normalize_target(self, acceleration: Tensor, *, accumulate: bool) -> Tensor

and the benchmark will score it through that instead. Its loss is then no longer
on the shared scale, which is the point, and the reported rollout metrics are
unaffected either way.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch_geometric.data import Data


__all__ = ["SimulatorModel", "integrate_acceleration", "predicted_acceleration", "current_velocity"]

#: Fields copied from the input graph onto the predicted frame unchanged.
_CARRIED = ("box", "box_tensor", "atom_ids", "atom_types", "molecule_ids")


@runtime_checkable
class SimulatorModel(Protocol):
    """What the benchmark requires of a model.

    `hyperparameters` is the dict that goes into the result cache key, so it must
    name every choice that changes the model's behaviour and nothing that does
    not. Anything about the *input* -- history length, which features, the graph
    radius -- belongs to the run's `InputGraphSpec`, not here.
    """

    hyperparameters: dict[str, Any]

    def __call__(self, graph: Data) -> Data:
        """Predict the next frame from one input graph."""
        ...


def current_velocity(graph: Data) -> Tensor:
    """`x_t - x_{t-1}` of the newest input frame, in sigma per frame."""
    return graph.pos - graph.prev_pos


def integrate_acceleration(graph: Data, acceleration: Tensor) -> Data:
    """Build the next frame from a predicted acceleration.

    Semi-implicit Euler at unit timestep, one dumped frame being the unit:

        x_{t+1} = x_t + (x_t - x_{t-1}) + a

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

    position = graph.pos + current_velocity(graph) + acceleration

    # Recompute the raw bond features at the new positions, under the minimum
    # image. Imported here to keep this module free of the graph machinery.
    from ..graph.build import minimum_image

    source, target = graph.bond_index
    vectors = minimum_image(position[source] - position[target], graph.box_tensor)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)

    frame = Data(
        x=position,
        pos=position,
        edge_index=graph.bond_index,
        edge_attr=torch.cat([vectors, lengths, graph.bond_attr[:, position.shape[1] + 1 :]], dim=1),
        bond_types=graph.bond_types,
        time=int(graph.time) + int(graph.frame_interval),
    )
    for field in _CARRIED:
        value = getattr(graph, field, None)
        if value is not None:
            setattr(frame, field, value)
    frame.acceleration = acceleration
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
    return frame.pos - graph.pos - current_velocity(graph)
