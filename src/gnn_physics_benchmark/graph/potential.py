"""The Kremer-Grest force field constants, as the graph and physics code see them.

`lj/cut` pair interactions plus `fene` bonds, i.e. what `potential.mod` writes
into the LAMMPS deck. The constants belong to the *dataset* -- they are what
generated the trajectories -- so a `KGPotential` is normally built with
`from_dataset`, not by hand.

The one knob here that is about the network rather than about the physics is
`graph_cutoff`, and it is on this object because the potential is already
threaded through every path that builds a graph. `cutoff` is the radius the force
field is defined at; `graph_cutoff` is the radius the message-passing graph is
built at. The graph radius is bounded above by the force-field cutoff: it may be
equal to it or smaller, never wider. An edge beyond the cutoff carries no
interaction at all, so widening the graph past it adds edges that correspond to
nothing physical. Pulling the radius in below the cutoff is a real tradeoff --
the GNS convention of a small radius with deep message passing, recovering over
several hops what the dropped edges carried -- and that is what `graph_cutoff`
is for.

Wherever a real energy, force or virial is summed the two must be the same,
because the sum runs over whatever edges the graph happens to carry and a
truncated neighbour list silently truncates the physics. Use `.physics` to get
the same force field with the full list restored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["KGPotential", "HarmonicPotential", "POTENTIALS", "from_dataset"]


@dataclass(frozen=True)
class KGPotential:
    """`pair_style lj/cut` + `bond_style fene`, evaluated on a graph.

    Defaults are the canonical Kremer-Grest parameters, `bond_coeff * 30.0 1.5
    1.0 1.0` and `pair_coeff * * 1.0 1.0 2.5`. `exclude_bonded` mirrors
    `special_bonds lj 0.0 1.0 1.0`: LAMMPS removes the pair interaction between
    directly bonded beads because the FENE bond already carries the repulsive
    core, and forgetting it double-counts that core on every bond.
    """

    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 2.5
    fene_k: float = 30.0
    fene_r0: float = 1.5
    skin: float = 0.0
    exclude_bonded: bool = True
    graph_cutoff: float | None = None

    def __post_init__(self) -> None:
        if self.graph_cutoff is not None and self.graph_cutoff > self.cutoff:
            raise ValueError(
                f"graph_cutoff {self.graph_cutoff} exceeds the force-field cutoff {self.cutoff}; "
                "the message-passing graph may be narrower than the force field but never wider, "
                "because an edge past the cutoff carries no interaction."
            )

    kind = "kremer_grest"
    has_pair_interaction = True

    @classmethod
    def from_dataset(cls, entry, *, graph_cutoff: float | None = None, skin: float = 0.0) -> "KGPotential":
        """The force field a dataset was generated with, at a chosen graph radius."""
        declared = dict(entry.potential or {})
        if declared.pop("kind", cls.kind) != cls.kind:
            raise ValueError(f"dataset {entry.key!r} does not declare a {cls.kind} potential.")
        return cls(**declared, graph_cutoff=graph_cutoff, skin=skin)

    @property
    def graph_radius(self) -> float:
        """The radius the message-passing graph is actually built at."""
        return self.cutoff if self.graph_cutoff is None else self.graph_cutoff

    @property
    def buffered_cutoff(self) -> float:
        return self.graph_radius + self.skin

    @property
    def truncates_graph(self) -> bool:
        """Whether the graph is narrower than the force field it represents."""
        return self.graph_cutoff is not None and self.graph_cutoff < self.cutoff

    @property
    def physics(self) -> "KGPotential":
        """The same force field with the full `cutoff` neighbour list restored."""
        return self if self.graph_cutoff is None else replace(self, graph_cutoff=None)


@dataclass(frozen=True)
class HarmonicPotential:
    """A network of harmonic springs, each with its own rest length.

        E = k (r - l0)^2 / 2,    F = -k (r - l0)   along the bond

    There is no pair interaction: a spring network's only edges are its bonds, so
    there is no neighbour list to build and nothing outside the bond graph
    contributes to an energy, a force or a virial.

    Where the stiffness comes from is declared per dataset, because the lab's
    spring-network sets do not agree on it:

    * `"column"` -- read straight from a named `edge_attr` column.
    * `"inverse_rest_length"` -- computed as `1 / l0`, the convention in which a
      spring cut from a uniform material has a stiffness inversely proportional
      to its length.

    Which of the two applies is not inferable from the numbers. In the
    `dePablo_random` set the stored column happens to satisfy `k = 1/l0**2`, and
    reading that as `1/l0` would give a force anti-aligned with the observed
    motion; in another set built to the second convention the same column would
    mean something else entirely. So the dataset says, and this class does not guess.
    """

    kind = "harmonic"
    has_pair_interaction = False

    columns: tuple[str, ...] = ()
    """The effective `edge_attr` column names of the dataset, for index lookup."""
    stiffness_from: str = "column"
    stiffness_column: str = "stiffness"
    rest_length_column: str = "rest_length"
    graph_cutoff: float | None = None

    @classmethod
    def from_dataset(cls, entry, *, graph_cutoff: float | None = None, skin: float = 0.0) -> "HarmonicPotential":
        declared = dict(entry.potential or {})
        declared.pop("kind", None)
        declared.pop("rest_length_from", None)  # only "first_frame" exists, and the loader applies it
        return cls(columns=tuple(entry.schema.edge_columns), graph_cutoff=graph_cutoff, **declared)

    @property
    def graph_radius(self) -> float:
        return 0.0

    @property
    def truncates_graph(self) -> bool:
        """A bond graph is never truncated: every bond is always present."""
        return False

    @property
    def physics(self) -> "HarmonicPotential":
        return self

    def rest_length(self, edge_attr):
        return edge_attr[:, self.columns.index(self.rest_length_column)]

    def stiffness(self, edge_attr):
        if self.stiffness_from == "column":
            return edge_attr[:, self.columns.index(self.stiffness_column)]
        return 1.0 / self.rest_length(edge_attr)


#: Registered potential kinds, by the `kind` a dataset declares.
POTENTIALS = {KGPotential.kind: KGPotential, HarmonicPotential.kind: HarmonicPotential}


def from_dataset(entry, *, graph_cutoff: float | None = None, skin: float = 0.0):
    """Build the potential a dataset declares, or `None` if it declares none."""
    if entry.potential is None:
        return None
    kind = entry.potential.get("kind", KGPotential.kind)
    if kind not in POTENTIALS:
        raise KeyError(f"dataset {entry.key!r} declares an unknown potential kind {kind!r}; known: {sorted(POTENTIALS)}.")
    return POTENTIALS[kind].from_dataset(entry, graph_cutoff=graph_cutoff, skin=skin)
