"""The input graph: what a model is actually handed, and how to ask for it.

A model never sees a raw frame. It sees one `Data` object built by the benchmark
from a window of `history + 1` consecutive frames, described by an
`InputGraphSpec`. The spec belongs to the run, not to the model: two models
compared on the same dataset key are handed exactly the same input, and a model
is told what it is getting rather than deciding it.

Why velocities and not positions
--------------------------------
The quantity to predict is the one-step acceleration, `(x_{t+1} - x_t) -
(x_t - x_{t-1})`. Absolute positions are O(3) sigma while that target is O(3e-5)
sigma, so feeding positions makes the network find a difference of five orders
in float32. Feeding the displacement history instead puts the input on the
target's own scale.

Affine and residual velocity
----------------------------
Compression is imposed by shrinking the box, so part of every bead's motion is
just the cell deforming under it. Splitting a step into

    affine   = x_t * (L_{t+1} / L_t - 1)      # what the box alone would do
    residual = (x_{t+1} - x_t) - affine       # what the material did on top

lets a run choose which of the two the network reads. `"residual"` hands it only
the non-affine part, which is the physically interesting one; `"total"` hands it
the raw displacement, i.e. `affine + residual`; `"affine"` hands it the box-driven
part alone, which is a diagnostic rather than a serious model input. Only the
selected one is attached -- the graph carries what the run asked for and nothing
else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch_geometric.data import Data


__all__ = ["InputGraphSpec", "INPUT_GRAPH_SCHEMA", "VELOCITY_MODES", "EDGE_MODES", "validate_input_graph"]

VELOCITY_MODES = ("total", "residual", "affine")
EDGE_MODES = ("bond+pair", "bond")


@dataclass(frozen=True)
class InputGraphSpec:
    """Everything about the input graph that a run gets to choose.

    The defaults are the sane starting point for somebody who only wants to plug
    in a model: an eight-step history of total velocity on a graph of FENE bonds
    plus a Lennard-Jones neighbour list pulled in to the first coordination shell.
    """

    dim: int = 3
    """Spatial dimension. Filled in from the dataset's `FrameSchema` when a run is
    resolved, so a run config need not repeat it; it is recorded in the saved
    config, and so in the result key, once resolved."""

    edge_width: int = 8
    """Number of `edge_attr` columns the model will see. Also filled in when the
    run is resolved: the dataset's own width under `edges="bond"`, and the typed
    bond-plus-pair width under `edges="bond+pair"`. A model needs it to size its
    edge encoder before it has seen a graph."""

    history: int = 8
    """Number of velocity steps fed to the model. Needs `history + 1` frames."""

    velocity: str = "total"
    """Which part of the displacement `x` carries: total, residual or affine."""

    fractional_coordinates: bool = False
    """Attach `pos / box_tensor`, an O(0.3) positional feature."""

    node_force: bool = False
    """Attach the analytic net force per particle. Available whenever the dataset
    declares a potential, including a bond-only spring network. Costs a
    full-cutoff neighbour list per frame when the graph is truncated; see
    `physics.node_force_feature`.

    Off by default, and how badly it behaves depends on the force law. For a
    harmonic network the force is linear in displacement, so it stays bounded and
    the feature is harmless. For a Lennard-Jones force field it goes as r^-13:
    two particles that drift close during a rollout produce an enormous input,
    the network answers with an enormous acceleration, and the rollout runs away.
    Measured on a small Kremer-Grest run, turning this on left the one-step
    training loss essentially unchanged (0.461 against 0.437) while the rollout
    diverged after six of fifteen steps. Treat it as a feature to test, not one
    to assume."""

    edges: str = "bond+pair"
    """`"bond+pair"` adds the Lennard-Jones neighbour list on top of the bonds
    and gives the typed 8-wide `edge_attr`. `"bond"` keeps only the symmetrised
    FENE bonds, which is what a dataset without a known pair potential would use."""

    graph_cutoff: float | None = None
    """Radius the pair neighbour list is built at, in sigma. `None`, the default,
    means the dataset's own force-field cutoff, which is the only choice that
    needs no justification.

    A smaller value is a real tradeoff and the reason this knob exists. For a
    full Lennard-Jones melt at 2.5 sigma every bead has about 68 neighbours --
    most of a 315-bead system -- so message passing reaches everything in two
    layers, depth buys nothing, and the graph costs about 21k edges. Pulling in
    to the first minimum of g(r), around 1.4-1.5 sigma, leaves about 12
    neighbours and 4.3k edges, and what the dropped edges carried is recovered
    over several hops instead.

    It may never be *larger* than the force-field cutoff: an edge past the cutoff
    carries no interaction, so a wider graph adds edges that correspond to nothing.
    `KGPotential` refuses one. For a purely repulsive WCA potential the cutoff is
    already the Lennard-Jones minimum at about 1.12 sigma, so `None` is the
    natural setting and there is little room to pull in further."""

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError(f"history must be at least 1, got {self.history}.")
        if self.velocity not in VELOCITY_MODES:
            raise ValueError(f"velocity must be one of {VELOCITY_MODES}, got {self.velocity!r}.")
        if self.edges not in EDGE_MODES:
            raise ValueError(f"edges must be one of {EDGE_MODES}, got {self.edges!r}.")
        if self.edges == "bond" and self.graph_cutoff is not None:
            raise ValueError(
                "edges='bond' builds no pair neighbour list, so graph_cutoff has nothing to "
                "act on; set it to None."
            )

    @property
    def window_length(self) -> int:
        """Frames needed to build one input graph."""
        return self.history + 1

    @property
    def node_feature_width(self) -> int:
        """Width of `x` on the resulting graph."""
        return self.history * self.dim

    @property
    def extra_node_channels(self) -> int:
        """Per-axis input channels beyond the velocity history."""
        return int(self.fractional_coordinates) + int(self.node_force)

    def resolved(self, schema) -> "InputGraphSpec":
        """This spec with `dim` and `edge_width` taken from a dataset's schema."""
        from dataclasses import replace

        width = schema.dim + 5 if self.edges == "bond+pair" else schema.edge_width
        if (self.dim, self.edge_width) == (schema.dim, width):
            return self
        return replace(self, dim=schema.dim, edge_width=width)

    def to_dict(self) -> dict:
        return asdict(self)


#: What the built graph carries. `h` is `spec.history`, `N` beads, `E` edges.
INPUT_GRAPH_SCHEMA = (
    ("x", "[N, 3h]", "the velocity history the spec selected, most recent step first, step-major"),
    ("pos", "[N, 3]", "positions of the newest input frame, i.e. x_t"),
    ("prev_pos", "[N, 3]", "positions of the frame before it, x_{t-1}; with `pos` this is "
                           "everything needed to integrate a predicted acceleration"),
    ("edge_index", "[2, E]", "symmetrised bonds, plus the pair neighbour list under edges='bond+pair'"),
    ("bond_index", "[2, B]", "the raw directed bond topology, for rebuilding a predicted frame"),
    ("bond_attr", "[B, 5]", "its raw 5-wide features, whose last column carries the FENE constant"),
    ("bond_types", "[B]", "backbone or crosslink, carried through to the predicted frame"),
    ("frame_interval", "scalar", "MD steps between consecutive frames of this window"),
    ("edge_attr", "[E, 8] or [E, 5]", "typed layout under 'bond+pair', raw bond layout under 'bond'"),
    ("fractional_coordinates", "[N, 3]", "pos / box_tensor; present only if requested"),
    ("node_force", "[N, 3]", "analytic net force per bead; present only if requested"),
    ("box", "-", "the Box of the newest input frame"),
    ("box_tensor", "[3]", "its edge lengths"),
    ("time", "scalar", "LAMMPS timestep of the newest input frame"),
    ("atom_ids", "[N]", "carried through unchanged"),
    ("atom_types", "[N]", "carried through unchanged"),
    ("molecule_ids", "[N]", "carried through unchanged"),
)


def validate_input_graph(graph: Data, spec: InputGraphSpec, *, name: str = "input graph") -> None:
    """Raise `ValueError` if `graph` does not match what `spec` asks for."""
    num_nodes = graph.pos.shape[0]
    expected = spec.node_feature_width
    if graph.x.shape != (num_nodes, expected):
        raise ValueError(f"{name}: expected x of shape [{num_nodes}, {expected}], got {tuple(graph.x.shape)}.")
    if graph.prev_pos.shape != (num_nodes, spec.dim):
        raise ValueError(
            f"{name}: expected prev_pos of shape [{num_nodes}, {spec.dim}], got {tuple(graph.prev_pos.shape)}."
        )

    if spec.edges == "bond+pair" and graph.edge_attr.shape[1] != spec.dim + 5:
        raise ValueError(
            f"{name}: edges='bond+pair' implies the typed edge_attr width of {spec.dim + 5}, "
            f"got {graph.edge_attr.shape[1]}."
        )
    for field, wanted in (("fractional_coordinates", spec.fractional_coordinates), ("node_force", spec.node_force)):
        present = getattr(graph, field, None) is not None
        if present != wanted:
            raise ValueError(f"{name}: spec asks for {field}={wanted} but the graph {'has' if present else 'lacks'} it.")
