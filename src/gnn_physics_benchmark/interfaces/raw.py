"""What a stored trajectory frame contains, declared per dataset.

A dataset is a directory of `.pt` files, one per simulated system, each
unpickling to a plain `list[Data]` -- one `torch_geometric.data.Data` per dumped
frame, in time order. Different datasets store different things in those frames,
so the layout is not a constant: every dataset registers a `FrameSchema` saying
what its frames hold, and everything downstream reads the schema rather than a
hard-coded dimension or column order.

What a schema has to pin down
-----------------------------
**The `edge_attr` columns, by name.** The benchmark needs only two facts about
them: which columns are *geometry*, recomputed whenever positions or the box
move, and which are *static parameters*, carried through unchanged. Naming the
columns gives both, and the spatial dimension falls out of how many vector
components there are:

    ("dx", "dy", "r", "stiffness")        -> 2D, one parameter per bond
    ("dx", "dy", "dz", "r", "fene_k")     -> 3D, one parameter per bond

The order is fixed: the vector components, then their length `r`, then any
parameters. `dim` is the number of vector components, so nothing anywhere needs
a `DIM` constant.

**Which fields are actually stored.** Everything else the benchmark relies on is
derived mechanically when a frame is loaded -- `pos` from `x`, `box_tensor` from
`box`, `time` from the frame index, identity fields from `arange`/`ones`. A
dataset that stores real values for those declares them and they are used as-is.

**Which module the `Box` was pickled from**, since that name is baked into the
file and differs between generations of the generating code.

Units are whatever the simulation used; both current datasets are in
Lennard-Jones reduced units. A "frame" is one dump, separated from the next by
`interval` simulation steps, not one timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

__all__ = [
    "FieldSpec",
    "FrameSchema",
    "DERIVED_COLUMN_RULES",
    "VECTOR_COLUMNS",
    "LENGTH_COLUMN",
    "ALWAYS_STORED",
    "DERIVABLE_FIELDS",
    "validate_raw_frame",
    "describe_schema",
]

#: Names reserved for the components of the edge vector, in order.
VECTOR_COLUMNS = ("dx", "dy", "dz")

#: Name reserved for the length of that vector.
LENGTH_COLUMN = "r"

#: Fields every raw frame must carry, whatever the dataset.
ALWAYS_STORED = ("x", "edge_index", "edge_attr", "box")

#: Fields the loader fills in when a dataset does not store them.
DERIVABLE_FIELDS = ("pos", "box_tensor", "time", "atom_ids", "atom_types", "molecule_ids", "bond_types")

#: Edge columns the loader can compute from a whole trajectory. `rest_length` is
#: the bond length in the first stored frame: these networks are generated at
#: their undeformed reference state, so every bond sits exactly at rest there.
#: Note the consequence -- a mechanical-equilibrium check on frame 0 is vacuous,
#: satisfied by any stiffness at all, so it identifies the rest length and says
#: nothing about the force law.
DERIVED_COLUMN_RULES = ("rest_length",)


@dataclass(frozen=True)
class FieldSpec:
    """One attribute of a raw frame."""

    name: str
    shape: str
    meaning: str
    stored: bool = True
    shared: bool = False
    """True when every frame of a trajectory reuses one allocation for this
    field. `torch.save` preserves that sharing, so copying frames naively can
    double a trajectory's memory; the window cache accounts for it."""


@dataclass(frozen=True)
class FrameSchema:
    """The layout of one dataset's stored frames."""

    edge_columns: tuple[str, ...]
    """`edge_attr` columns in order: the vector components, then `r`, then any
    per-edge parameters."""

    derived_columns: tuple[str, ...] = ()
    """Edge columns appended at load time rather than read from the file, named
    from the tail of `edge_columns`. The only rule implemented is `rest_length`:
    the bond length in the trajectory's first frame, which for a network built at
    its undeformed reference state is the rest length of every bond."""

    stored_fields: tuple[str, ...] = ()
    """Fields from `DERIVABLE_FIELDS` that this dataset really stores. Anything
    not listed is derived at load time."""

    pickle_module: str = "network_minimal"
    """Module the `Box` instances in the `.pt` files were pickled from."""

    column_docs: dict = field(default_factory=dict)
    """Optional prose per edge column, for `gnn-bench schema`."""

    def __post_init__(self) -> None:
        columns = self.edge_columns
        vectors = [name for name in columns if name in VECTOR_COLUMNS]
        if not vectors:
            raise ValueError(f"edge_columns {columns} names no vector component from {VECTOR_COLUMNS}.")
        if tuple(columns[: len(vectors)]) != tuple(VECTOR_COLUMNS[: len(vectors)]):
            raise ValueError(
                f"edge_columns must start with {VECTOR_COLUMNS[: len(vectors)]} in order, got {columns}."
            )
        if len(columns) <= len(vectors) or columns[len(vectors)] != LENGTH_COLUMN:
            raise ValueError(f"edge_columns must put {LENGTH_COLUMN!r} directly after the vector, got {columns}.")
        unknown = set(self.stored_fields) - set(DERIVABLE_FIELDS)
        if unknown:
            raise ValueError(f"stored_fields names {sorted(unknown)}, which are not in {DERIVABLE_FIELDS}.")
        if self.derived_columns and tuple(columns[-len(self.derived_columns) :]) != tuple(self.derived_columns):
            raise ValueError(
                f"derived_columns {self.derived_columns} must be the last entries of edge_columns {columns}."
            )
        unknown = set(self.derived_columns) - set(DERIVED_COLUMN_RULES)
        if unknown:
            raise ValueError(f"no rule for derived edge column(s) {sorted(unknown)}.")

    @property
    def dim(self) -> int:
        """Spatial dimension, from how many vector components the edges carry."""
        return sum(name in VECTOR_COLUMNS for name in self.edge_columns)

    @property
    def edge_width(self) -> int:
        """Columns a loaded frame carries, derived ones included."""
        return len(self.edge_columns)

    @property
    def stored_edge_width(self) -> int:
        """Columns actually present in the `.pt` file."""
        return len(self.edge_columns) - len(self.derived_columns)

    @property
    def vector_slice(self) -> slice:
        return slice(0, self.dim)

    @property
    def length_index(self) -> int:
        return self.dim

    @property
    def parameter_slice(self) -> slice:
        """Columns carried through unchanged when the geometry is recomputed."""
        return slice(self.dim + 1, self.edge_width)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.edge_columns[self.dim + 1 :])

    def column(self, name: str) -> int:
        """Index of a named edge column."""
        if name not in self.edge_columns:
            raise KeyError(f"no edge column {name!r}; this dataset has {self.edge_columns}.")
        return self.edge_columns.index(name)

    def stores(self, name: str) -> bool:
        return name in self.stored_fields


def validate_raw_frame(frame: Data, schema: FrameSchema, *, name: str = "frame") -> None:
    """Raise `ValueError` if `frame` does not match `schema`.

    Checks the things that silently corrupt results if they are wrong -- a
    missing field, a mismatched node or edge count, the wrong `edge_attr` width,
    `x` and `pos` having drifted apart. This is the one place the benchmark
    validates data: it runs when a dataset is registered and when a model's
    output is checked, not inside the training loop.
    """
    present = set(frame.keys())
    missing = [wanted for wanted in ALWAYS_STORED if wanted not in present]
    if missing:
        raise ValueError(f"{name}: missing required field(s) {missing}. Present: {sorted(present)}.")

    dim = schema.dim
    num_nodes = frame.x.shape[0]
    if frame.x.shape != (num_nodes, dim):
        raise ValueError(f"{name}: expected x of shape [N, {dim}], got {tuple(frame.x.shape)}.")

    num_edges = frame.edge_index.shape[1]
    if frame.edge_index.shape[0] != 2:
        raise ValueError(f"{name}: expected edge_index of shape [2, E], got {tuple(frame.edge_index.shape)}.")
    if frame.edge_attr.shape != (num_edges, schema.edge_width):
        raise ValueError(
            f"{name}: schema {schema.edge_columns} implies edge_attr of shape "
            f"[{num_edges}, {schema.edge_width}], got {tuple(frame.edge_attr.shape)}."
        )
    if int(frame.edge_index.max()) >= num_nodes:
        raise ValueError(f"{name}: edge_index refers to node {int(frame.edge_index.max())} of {num_nodes}.")

    if "pos" in present:
        if frame.pos.shape != frame.x.shape:
            raise ValueError(f"{name}: pos {tuple(frame.pos.shape)} does not match x {tuple(frame.x.shape)}.")
        if not torch.equal(frame.x, frame.pos):
            raise ValueError(f"{name}: x and pos must hold the same positions on a raw frame.")
    if "box_tensor" in present and frame.box_tensor.shape != (dim,):
        raise ValueError(f"{name}: expected box_tensor of shape [{dim}], got {tuple(frame.box_tensor.shape)}.")
    for one_per_node in ("atom_ids", "atom_types", "molecule_ids"):
        if one_per_node in present and frame[one_per_node].shape != (num_nodes,):
            raise ValueError(
                f"{name}: expected {one_per_node} of shape [{num_nodes}], got {tuple(frame[one_per_node].shape)}."
            )
    if "bond_types" in present and frame.bond_types.shape != (num_edges,):
        raise ValueError(f"{name}: expected bond_types of shape [{num_edges}], got {tuple(frame.bond_types.shape)}.")


#: Documentation for the fields that are common to every dataset.
COMMON_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("x", "[N, dim]", "Unwrapped absolute positions. Unwrapped means the periodic image is "
              "tracked across frames, so a frame-to-frame difference is a real displacement and never "
              "jumps by a box length. Row i is the same particle in every frame."),
    FieldSpec("pos", "[N, dim]", "A clone of `x`. Parts of torch_geometric and of the physics code read "
              "`pos` while the encoders read `x`; on a raw frame the two are always equal.", stored=False),
    FieldSpec("edge_index", "[2, E]", "Bond connectivity, directed: each bond appears once, and no "
              "non-bonded pair appears at all. The input-graph builder symmetrises it and may add a pair "
              "neighbour list; a model never sees this raw edge set.", shared=True),
    FieldSpec("edge_attr", "[E, W]", "Per-bond features; see the dataset's edge columns below."),
    FieldSpec("box", "-", "A `gnn_physics_benchmark.interfaces.box.Box`: the periodic bounds. Pickled "
              "into the file, so the class must be importable to load a trajectory."),
    FieldSpec("box_tensor", "[dim]", "The box edge lengths as a tensor that can move to a device. This is "
              "the field all periodic-boundary and strain code uses. Deformation is imposed by changing it.",
              stored=False),
    FieldSpec("time", "scalar", "Simulation step this frame was dumped at. Consecutive frames differ by "
              "the dataset's `interval`.", stored=False),
    FieldSpec("atom_ids", "[N]", "Particle ids, constant across the trajectory.", stored=False, shared=True),
    FieldSpec("atom_types", "[N]", "Per-particle type. Constant.", stored=False, shared=True),
    FieldSpec("molecule_ids", "[N]", "Which molecule or chain each particle belongs to. Constant.",
              stored=False, shared=True),
    FieldSpec("bond_types", "[E]", "Per-bond type. Indexes the *directed* edge_index, so it is dropped "
              "when the graph is symmetrised.", stored=False, shared=True),
)


def describe_schema(schema: FrameSchema) -> str:
    """One dataset's frame layout as a printable table."""
    lines = [f"dim {schema.dim}   edge_attr width {schema.edge_width}   Box pickled from {schema.pickle_module!r}", ""]
    header = f"{'field':14s} {'shape':12s} {'source':9s} meaning"
    lines += [header, "-" * len(header)]
    for spec in COMMON_FIELDS:
        stored = spec.stored or schema.stores(spec.name)
        shape = spec.shape.replace("dim", str(schema.dim)).replace("W", str(schema.edge_width))
        lines.append(f"{spec.name:14s} {shape:12s} {'stored' if stored else 'derived':9s} {spec.meaning}")
    lines += ["", "edge_attr columns:"]
    for index, name in enumerate(schema.edge_columns):
        if name in VECTOR_COLUMNS:
            role = "geometry, recomputed when positions or the box move"
        elif name == LENGTH_COLUMN:
            role = "geometry, the length of that vector"
        else:
            role = "parameter, carried through unchanged"
        note = schema.column_docs.get(name, "")
        lines.append(f"  [{index}] {name:12s} {role}{'  -- ' + note if note else ''}")
    return "\n".join(lines)
