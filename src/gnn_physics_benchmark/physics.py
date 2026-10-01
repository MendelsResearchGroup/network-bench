"""Energy, forces and virial stress of a Kremer-Grest melt, evaluated on a graph.

Two force fields are implemented, dispatched on the `kind` a dataset declares:
a Kremer-Grest melt and a harmonic spring network. Everything below the dispatch
-- net forces, the virial, the stress -- is written once and works for either,
because all it needs from a potential is the radial force on each edge.

The Kremer-Grest force field is `bond_style fene` + `pair_style lj/cut`:

    E_bond = -K R0^2 / 2 * ln[1 - (r / R0)^2] + E_wca(r)

where `E_wca` is the shifted repulsive core LAMMPS always adds on top of a FENE
bond. The spring is singular as `r -> R0` rather than linear, which is exactly
what makes the chains uncrossable, so `r` is clamped just below `R0` instead of
being allowed to produce a NaN when a rollout overshoots.

Units are LAMMPS `lj` throughout, where `kB = 1` by definition. The virial
stress is returned as `[xx, yy, zz, xy, xz, yz]` in epsilon/sigma^3.

Every function here takes the typed 8-wide edge layout on a bidirectional graph,
and the `0.5` factors undo that double counting.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from .graph.build import IS_BOND, minimum_image, rebuilt_edges
from .graph.potential import HarmonicPotential, KGPotential

__all__ = ["WCA_FACTOR", "edge_forces", "node_force_feature", "virial_stress"]

#: Where `pair_style lj/cut` turns purely repulsive: 2^(1/6) sigma.
WCA_FACTOR = 2.0 ** (1.0 / 6.0)


def _harmonic_terms(dist: Tensor, edge_attr: Tensor, force_field: HarmonicPotential):
    """`(energy, radial force)` of a harmonic spring. Negative force is attractive.

    `E = k (r - l0)^2 / 2`, so `F = -k (r - l0)`: a stretched spring pulls its
    ends together, which is a negative magnitude in the sign convention here
    (positive is repulsive, as for the Lennard-Jones core).
    """
    l0 = force_field.rest_length(edge_attr)
    k = force_field.stiffness(edge_attr)
    extension = dist - l0
    return 0.5 * k * extension**2, -k * extension


def _require_full_graph(force_field) -> None:
    """Refuse to sum a physical quantity over a deliberately truncated graph.

    Every function here sums over the edges the graph happens to carry, so a
    neighbour list built at a shorter `graph_cutoff` silently truncates the sum
    rather than producing a slightly different number: the attractive tail is
    where most of the pair energy lives. The caller wants `force_field.physics`
    and a graph rebuilt from it.
    """
    if force_field.truncates_graph:
        raise ValueError(
            f"this force field builds its graph at {force_field.graph_radius} sigma but is "
            f"defined at {force_field.cutoff} sigma, so a sum over its edges would be "
            "truncated. Pass `force_field.physics` and a graph rebuilt from it."
        )


def _geometry(graph: Data) -> tuple[Tensor, Tensor, Tensor]:
    """`(distance, unit_vector, edge_vector)` for every edge, under PBC."""
    pos = graph.pos if graph.pos is not None else graph.x
    sender, receiver = graph.edge_index
    vec = minimum_image(pos[sender] - pos[receiver], graph.box_tensor)
    dist = torch.linalg.vector_norm(vec, dim=1)
    safe = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
    return dist, vec / safe.unsqueeze(1), vec


def _unpack(graph: Data) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """`(is_bond, distance, unit_vector, edge_vector, p1, p2)` for every edge of a typed graph."""
    dist, unit, vec = _geometry(graph)
    edge_attr = graph.edge_attr
    return edge_attr[:, IS_BOND].bool(), dist, unit, vec, edge_attr[:, 6], edge_attr[:, 7]


def _lj_terms(dist: Tensor, epsilon: Tensor, sigma: Tensor, cutoff: float | None):
    """`(energy, radial force)` of a Lennard-Jones pair, zeroed past `cutoff`."""
    safe = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
    sr6 = (sigma / safe) ** 6
    sr12 = sr6**2

    energy = 4.0 * epsilon * (sr12 - sr6)
    force = (24.0 * epsilon / safe) * (2.0 * sr12 - sr6)

    if cutoff is not None:
        inside = dist < cutoff
        # `pair_modify shift yes`: the energy is shifted to zero at the cutoff.
        sr6_c = (sigma / cutoff) ** 6
        energy = torch.where(inside, energy - 4.0 * epsilon * (sr6_c**2 - sr6_c), torch.zeros_like(energy))
        force = torch.where(inside, force, torch.zeros_like(force))

    return energy, force


def _fene_terms(dist: Tensor, k: Tensor, r0: Tensor, force_field: KGPotential):
    """`(energy, radial force)` of a LAMMPS FENE bond, repulsive core included."""
    clamped = torch.minimum(dist, 0.999 * r0)
    ratio = (clamped / r0) ** 2

    energy = -0.5 * k * r0**2 * torch.log1p(-ratio)
    force = -k * clamped / (1.0 - ratio)

    # The core is a WCA potential: an LJ cut and shifted at 2^(1/6) sigma, which
    # is exactly the `+ epsilon` that LAMMPS writes in the FENE bond formula.
    epsilon = torch.full_like(dist, force_field.epsilon)
    sigma = torch.full_like(dist, force_field.sigma)
    core_energy, core_force = _lj_terms(clamped, epsilon, sigma, WCA_FACTOR * force_field.sigma)

    return energy + core_energy, force + core_force


def _edge_terms(graph: Data, force_field) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`(energy, radial force, unit vector, edge vector)` per edge, for any potential.

    This is the single place a force field is interpreted; every physical
    quantity below is built from what it returns. Positive force is repulsive.
    """
    if force_field.kind == "harmonic":
        dist, unit, vec = _geometry(graph)
        energy, force = _harmonic_terms(dist, graph.edge_attr, force_field)
        return energy, force, unit, vec

    is_bond, dist, unit, vec, p1, p2 = _unpack(graph)
    bond_energy, bond_force = _fene_terms(dist, p1, p2, force_field)
    pair_energy, pair_force = _lj_terms(dist, p1, p2, force_field.cutoff)
    return (
        torch.where(is_bond, bond_energy, pair_energy),
        torch.where(is_bond, bond_force, pair_force),
        unit,
        vec,
    )


def edge_forces(graph: Data, force_field) -> tuple[Tensor, Tensor]:
    """`(radial force magnitude, edge vector)` per edge. Positive is repulsive."""
    _require_full_graph(force_field)
    _, force, _, vec = _edge_terms(graph, force_field)
    return force, vec


def _net_force(graph: Data, force_field) -> Tensor:
    """Net force per particle summed over whatever edges `graph` carries.

    The graph is bidirectional, so every interacting pair appears twice with the
    roles swapped and accumulating on the receiver alone covers both ends.
    """
    _, magnitude, unit, _ = _edge_terms(graph, force_field)
    pos = graph.pos if graph.pos is not None else graph.x
    receiver = graph.edge_index[1]
    return torch.zeros_like(pos).index_add(0, receiver, -magnitude.unsqueeze(1) * unit)


def node_force_feature(graph: Data, force_field) -> Tensor:
    """The same sum, offered to the network as an input feature, `[N, 3]`.

    The point of the feature is that a network should not have to rediscover
    `r^-13` from a list of distances when the force law is known in closed form.

    The full neighbour list is rebuilt first when the graph is truncated, and
    that is not an optional nicety. The cheap version looks reasonable and is
    not: this ensemble sits at essentially zero temperature, so every frame is at
    mechanical equilibrium and the net force per bead at the full 2.5 sigma is
    about 2.3e-4, five orders below the 1.07 of a single pair at contact. That
    number is small because the shells cancel. Summing over a 1.5 sigma graph
    destroys the cancellation rather than approximating it -- the result is about
    1.07 per bead, some 5000x the real force, correlating with the true force at
    0.003. It is not a truncated force, it is the truncation.

    So the feature costs a second, full-cutoff neighbour list per frame whenever
    `graph_cutoff` is set. There is a second price too: a residual of 2e-4
    reached by cancelling ~68 edge forces of up to 20 each is not something
    float32 does cleanly. Against a float64 evaluation of the same frames the
    float32 result keeps the magnitude but only about 0.88 of the direction. Use
    the feature knowing that roughly a tenth of it is rounding.
    """
    full = force_field.physics
    if full is force_field:
        return _net_force(graph, force_field)

    index, attr = rebuilt_edges(graph, force_field=full)
    restored = Data(
        x=graph.x,
        pos=graph.pos,
        edge_index=index,
        edge_attr=attr,
        box_tensor=graph.box_tensor,
    )
    return _net_force(restored, full)


def virial_stress(graph: Data, force_field) -> Tensor:
    """Virial stress as `[xx, yy, zz, xy, xz, yz]`, in epsilon / sigma^3."""
    magnitude, vec = edge_forces(graph, force_field)

    dist = torch.linalg.vector_norm(vec, dim=1)
    safe = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
    force_vec = (vec / safe.unsqueeze(1)) * magnitude.unsqueeze(1)

    volume = torch.prod(graph.box_tensor)
    dim = graph.box_tensor.numel()
    # Diagonal first, then the off-diagonals: [xx, yy, xy] in 2D,
    # [xx, yy, zz, xy, xz, yz] in 3D. "Volume" is an area in two dimensions.
    pairs = [(a, a) for a in range(dim)] + [(a, b) for a in range(dim) for b in range(a + 1, dim)]
    return torch.stack([0.5 * torch.sum(force_vec[:, a] * vec[:, b]) / volume for a, b in pairs])
