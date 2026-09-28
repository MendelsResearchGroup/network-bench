"""Assembling the input graph a model is run on, from a window of raw frames.

Two steps, kept apart because the training loop caches the first one:

* `prepare_window` takes `history + 1` raw frames and returns prepared graphs --
  symmetrised, with the pair neighbour list on the newest frame only. The earlier
  frames are read for their positions alone, so building a neighbour list on them
  would be most of the cost for none of the benefit.
* `build_input_graph` folds that window into the single `Data` the model sees.

The split between affine and residual velocity is the substantive part. See
`gnn_physics_benchmark.interfaces.graph` for what the two mean and why the run,
not the model, chooses between them.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from ..interfaces.graph import InputGraphSpec
from .build import prepare_frames
from . import potential as potentials
from .potential import KGPotential

__all__ = ["affine_residual_velocity", "prepare_window", "build_input_graph"]

#: Fields copied straight from the newest input frame onto the input graph.
#: `bond_*` is the raw bond topology, which `integrate_acceleration` needs to
#: return a predicted frame in the dataset's own schema.
CARRIED = (
    "box",
    "box_tensor",
    "time",
    "atom_ids",
    "atom_types",
    "molecule_ids",
    "bond_index",
    "bond_attr",
    "bond_types",
)


def affine_residual_velocity(earlier: Data, later: Data) -> tuple[Tensor, Tensor]:
    """Split the step from `earlier` to `later` into box-driven and material parts.

    The affine part is what a uniform deformation of the cell would have done to
    a bead sitting where it was; the residual is everything the material did on
    top of that.
    """
    scale = later.box_tensor / earlier.box_tensor
    affine = earlier.x * (scale - 1.0)
    residual = (later.x - earlier.x) - affine
    return affine, residual


def potential_for(spec: InputGraphSpec, entry):
    """The force field for a run on a given dataset, or `None` if it declares none.

    Dispatched on the `kind` the dataset registered, so a spring network and a
    Kremer-Grest melt are both served without either knowing about the other. A
    potential with no pair interaction is still returned under `edges="bond"`,
    because the stress metrics and the analytic force need it even though there
    is no neighbour list to build.
    """
    if entry.potential is None:
        return None
    built = potentials.from_dataset(entry, graph_cutoff=spec.graph_cutoff)
    if spec.edges == "bond+pair" and not built.has_pair_interaction:
        raise ValueError(
            f"dataset {entry.key!r} declares a {built.kind!r} potential with no pair interaction, "
            "so there is no neighbour list to add; use edges='bond'."
        )
    return built


def prepare_window(
    frames: list[Data], spec: InputGraphSpec, potential: KGPotential | None
) -> list[Data]:
    """Symmetrise a window of raw frames and give the newest one its pair edges."""
    if len(frames) != spec.window_length:
        raise ValueError(
            f"spec.history={spec.history} needs a window of {spec.window_length} frames, got {len(frames)}."
        )
    return prepare_frames(frames, potential, pair_edges_on="last")


def build_input_graph(
    window: list[Data], spec: InputGraphSpec, potential: KGPotential | None = None
) -> Data:
    """Fold a prepared window into the single graph the model is handed.

    `window` is in time order, oldest first; the newest frame supplies the graph
    structure, the positions and the box, and the whole window supplies the
    velocity history.
    """
    if len(window) != spec.window_length:
        raise ValueError(
            f"spec.history={spec.history} needs a window of {spec.window_length} frames, got {len(window)}."
        )

    base = window[-1]

    # Most recent step first, so that `x.view(N, history, 3)` indexes history
    # step 0 as the latest one. The encoders rely on this layout.
    steps = []
    for index in range(len(window) - 1, 0, -1):
        affine, residual = affine_residual_velocity(window[index - 1], window[index])
        steps.append(
            {"total": affine + residual, "residual": residual, "affine": affine}[spec.velocity]
        )
    selected = torch.cat(steps, dim=1)

    graph = Data(
        x=selected,
        pos=base.x,
        prev_pos=window[-2].x,
        edge_index=base.edge_index,
        edge_attr=base.edge_attr,
    )
    for field in CARRIED:
        value = getattr(base, field, None)
        if value is not None:
            setattr(graph, field, value)
    # Frames are separated by a fixed number of MD steps; a predicted frame
    # continues the same clock. Read off the window rather than the dataset, so
    # that a run using `frame_stride` gets the stride it actually loaded.
    graph.frame_interval = int(base.time) - int(window[-2].time)

    if spec.fractional_coordinates:
        graph.fractional_coordinates = base.x / base.box_tensor
    if spec.node_force:
        if potential is None:
            raise ValueError("node_force needs the dataset's force field; pass `potential`.")
        from ..physics import node_force_feature

        graph.node_force = node_force_feature(base, potential)

    return graph
