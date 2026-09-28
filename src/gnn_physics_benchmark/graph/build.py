"""Turning raw, bond-only frames into the graph a model is run on.

A stored frame carries one directed edge per FENE bond and nothing else. The
model's graph is built from it in two moves:

1. **Symmetrise.** Each bond becomes a pair of half-edges with the edge vector
   flipped on the reverse one, so message passing can run in both directions.
   `bond_types` is deliberately dropped here: it indexes the directed bond list,
   and symmetrising changes the edge set. The backbone/crosslink distinction
   survives on the nodes, in `atom_types`.
2. **Add the pair neighbour list.** Every non-bonded pair within the graph radius
   becomes an edge. This promotes `edge_attr` from the 5-wide raw layout to the
   8-wide typed layout, which is what a model actually sees.

Edge feature layouts
--------------------
The **raw** layout is the dataset's own, declared by its `FrameSchema`: the edge
vector, its length, then the per-bond parameters, e.g. `[dx, dy, dz, r, fene_k]`
in three dimensions or `[dx, dy, r, stiffness]` in two.

The **typed** layout, `typed_width(dim) = dim + 5`, is what a bond-plus-pair graph
carries::

    [is_bond, is_pair, dx, dy, dz, r, p1, p2]

with two parameter slots, because a bond and a pair edge carry different
constants: `(K, R0)` against `(epsilon, sigma)` for Kremer-Grest. A dataset with
no pair interaction never reaches this layout and stays on the raw one.

A note on periodic boundaries: the box of this ensemble is about 6.6 sigma, so a
2.5 sigma cutoff with any real buffer already approaches half the box edge.
`check_minimum_image` refuses a cutoff that breaks the convention rather than
letting a wrapped displacement quietly become wrong.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected

from .potential import KGPotential

__all__ = [
    "bond_width",
    "typed_width",
    "IS_BOND",
    "IS_PAIR",
    "TYPED_EDGE_COLUMNS",
    "infer_dim",
    "spatial_slice",
    "minimum_image",
    "check_minimum_image",
    "add_box_tensor",
    "to_undirected_graph",
    "append_pair_interactions",
    "bond_edges",
    "pair_edges",
    "rebuilt_edges",
    "prepare_frames",
]

def bond_width(dim: int) -> int:
    """Width of a raw bond-only `edge_attr` carrying a single parameter."""
    return dim + 2


def typed_width(dim: int) -> int:
    """Width of the typed bond-plus-pair `edge_attr`."""
    return dim + 5

#: Column of the one-hot flag marking a bond (as opposed to a pair) edge.
IS_BOND = 0
IS_PAIR = 1

#: The typed layout, documented column by column.
TYPED_EDGE_COLUMNS = (
    ("is_bond", "1.0 on a FENE bond edge, 0.0 otherwise"),
    ("is_pair", "1.0 on a non-bonded Lennard-Jones edge, 0.0 otherwise"),
    ("dx", "x component of the minimum-image edge vector, source minus target, in sigma"),
    ("dy", "y component of the same vector"),
    ("dz", "z component of the same vector"),
    ("r", "length of that vector, in sigma"),
    ("K_or_epsilon", "FENE spring constant on a bond edge, LJ epsilon on a pair edge"),
    ("R0_or_sigma", "FENE maximum extent on a bond edge, LJ sigma on a pair edge"),
)


def infer_dim(graph: Data) -> int:
    """Spatial dimension of a graph, from the box if it has one."""
    box_tensor = getattr(graph, "box_tensor", None)
    if isinstance(box_tensor, Tensor):
        return int(box_tensor.numel())
    return int(graph.pos.shape[1]) if graph.pos is not None else int(graph.x.shape[1])


def spatial_slice(num_edge_features: int, dim: int) -> slice:
    """Columns of `edge_attr` holding the edge vector, for either layout.

    A raw layout leads with the vector; the typed one leads with the two one-hot
    type flags. The number of trailing parameter columns is the dataset's own, so
    the two are told apart by the typed width rather than by counting parameters.
    """
    if num_edge_features == typed_width(dim):
        return slice(2, 2 + dim)
    if num_edge_features > dim:
        return slice(0, dim)
    raise ValueError(f"edge_attr width {num_edge_features} is too narrow to hold a {dim}D vector.")


def minimum_image(dr: Tensor, box_tensor: Tensor) -> Tensor:
    """Wrap displacements into the primary image. Works on any trailing shape."""
    box = box_tensor.view(*([1] * (dr.dim() - 1)), -1)
    return dr - torch.round(dr / box) * box


def check_minimum_image(cutoff: float, box_tensor: Tensor) -> None:
    half = float(box_tensor.detach().min()) / 2.0
    if cutoff > half:
        raise ValueError(
            f"cutoff {cutoff:.3f} exceeds half the shortest box edge ({half:.3f}); "
            "the minimum-image convention does not hold. Lower the cutoff or skin, "
            "or use a larger system."
        )


def add_box_tensor(graph: Data) -> Data:
    """Attach `[Lx, Ly, Lz]` as a tensor, so the box survives `.to(device)`."""
    graph.box_tensor = torch.tensor(
        [graph.box.x, graph.box.y, graph.box.z], device=graph.x.device, dtype=graph.x.dtype
    )
    return graph


def _edge_vectors(graph: Data) -> Tensor:
    source, target = graph.edge_index[0], graph.edge_index[1]
    return minimum_image(graph.x[source] - graph.x[target], graph.box_tensor)


def to_undirected_graph(graph: Data) -> Data:
    """Symmetrise a directed physics graph, flipping the edge vectors.

    Scalars -- lengths, K, R0, epsilon, sigma, the one-hot flags -- are identical
    on the two half-edges, so the `mean` reduction of `coalesce` leaves them alone.
    """
    if is_undirected(graph.edge_index):
        return graph

    dim = infer_dim(graph)
    row, col = graph.edge_index
    reversed_index = torch.stack([col, row], dim=0)
    vector_columns = spatial_slice(graph.edge_attr.shape[1], dim)
    reversed_attr = graph.edge_attr.clone()
    reversed_attr[:, vector_columns] = -reversed_attr[:, vector_columns]

    edge_index, edge_attr = coalesce(
        torch.cat([graph.edge_index, reversed_index], dim=-1),
        torch.cat([graph.edge_attr, reversed_attr], dim=0),
        num_nodes=graph.num_nodes,
        reduce="mean",
    )

    out = Data(
        x=graph.x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        box=getattr(graph, "box", None),
        box_tensor=getattr(graph, "box_tensor", None),
    )
    for field in ("time", "atom_ids", "atom_types", "molecule_ids"):
        value = getattr(graph, field, None)
        if value is not None:
            setattr(out, field, value)
    return out


def bond_edges(graph: Data, force_field: KGPotential | None = None) -> tuple[Tensor, Tensor]:
    """Recompute the FENE bond edges of a typed graph at the current positions.

    `(K, R0)` is carried over from the input unless `force_field` is given.
    """
    dim = infer_dim(graph)
    if graph.edge_attr.shape[1] != dim + 5:
        raise ValueError(f"expected the typed layout of width {dim + 5}, got {graph.edge_attr.shape[1]}.")

    mask = graph.edge_attr[:, IS_BOND].bool()
    attr = graph.edge_attr[mask]
    edge_index = graph.edge_index[:, mask]

    source, target = edge_index
    vectors = minimum_image(graph.x[source] - graph.x[target], graph.box_tensor)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)

    if force_field is None:
        params = attr[:, dim + 3 :]
    else:
        params = torch.tensor(
            [[force_field.fene_k, force_field.fene_r0]], device=attr.device, dtype=attr.dtype
        ).repeat(attr.shape[0], 1)

    return edge_index, torch.cat([attr[:, 0:2], vectors, lengths, params], dim=-1)


def pair_edges(
    graph: Data, force_field: KGPotential, bonded_pairs: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Build the `lj/cut` pair edges at the current positions, under PBC.

    `bonded_pairs` is a `[2, E]` index of the 1-2 neighbours to leave out; pass
    `None` to look them up from the typed `edge_attr` of `graph` itself. The
    returned edges are bidirectional by construction.

    The neighbour list is a dense `N x N` distance matrix. At 315 beads that is
    cheap and exact; it is the reason a larger system would need a cell list.
    """
    device = graph.x.device
    box = graph.box_tensor
    cutoff = force_field.buffered_cutoff
    check_minimum_image(cutoff, box)

    dr = minimum_image(graph.x.unsqueeze(1) - graph.x.unsqueeze(0), box)
    distance = torch.norm(dr, dim=-1)
    mask = (distance < cutoff) & (distance > 1e-6)

    if force_field.exclude_bonded:
        if bonded_pairs is None:
            bonded_pairs = graph.edge_index[:, graph.edge_attr[:, IS_BOND].bool()]
        row, col = bonded_pairs
        mask[row, col] = False
        mask[col, row] = False

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    num_edges = edge_index.shape[1]

    one_hot = torch.tensor([[0.0, 1.0]], device=device, dtype=graph.x.dtype).repeat(num_edges, 1)
    vectors = dr[row, col]
    lengths = distance[row, col].unsqueeze(-1)
    epsilon = torch.full((num_edges, 1), force_field.epsilon, device=device, dtype=graph.x.dtype)
    sigma = torch.full((num_edges, 1), force_field.sigma, device=device, dtype=graph.x.dtype)

    return edge_index, torch.cat([one_hot, vectors, lengths, epsilon, sigma], dim=-1)


def rebuilt_edges(
    graph: Data, force_field: KGPotential | None = None
) -> Tensor | tuple[Tensor, Tensor]:
    """Recompute edge features after the positions or the box have moved.

    Returns a bare `edge_attr` for a bond-only graph, whose topology is fixed,
    and an `(edge_index, edge_attr)` pair for a typed graph, whose pair
    neighbour list is not. This is what a rollout calls after every predicted
    step; it is also the reason `graph_cutoff` matters, since the list is rebuilt
    at `force_field.buffered_cutoff`.
    """
    dim = infer_dim(graph)
    width = graph.edge_attr.shape[1]

    if width == dim + 2:
        vectors = _edge_vectors(graph)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        return torch.cat([vectors, lengths, graph.edge_attr[:, dim + 1 :]], dim=-1)

    if width == dim + 5:
        if force_field is None:
            raise ValueError("a typed graph needs a `force_field` to rebuild its pair edges.")
        bonded_pairs = graph.edge_index[:, graph.edge_attr[:, IS_BOND].bool()]
        bond_index, bond_attr = bond_edges(graph)
        pair_index, pair_attr = pair_edges(graph, force_field, bonded_pairs=bonded_pairs)
        return (
            torch.cat([bond_index, pair_index], dim=-1),
            torch.cat([bond_attr, pair_attr], dim=0),
        )

    raise ValueError(f"unexpected edge_attr width {width} for dim {dim}.")


def append_pair_interactions(graph: Data, force_field: KGPotential) -> Data:
    """Promote a bond-only graph to the typed layout and add the pair edges.

    Used once, when a window is prepared; afterwards `rebuilt_edges` keeps the
    pair neighbour list up to date.
    """
    dim = infer_dim(graph)
    if graph.edge_attr.shape[1] != dim + 2:
        raise ValueError(f"expected the bond-only layout of width {dim + 2}, got {graph.edge_attr.shape[1]}.")

    device, dtype = graph.x.device, graph.x.dtype
    num_bonds = graph.edge_index.shape[1]

    bond_one_hot = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(num_bonds, 1)
    fene_r0 = torch.full((num_bonds, 1), force_field.fene_r0, device=device, dtype=dtype)
    bond_attr = torch.cat([bond_one_hot, graph.edge_attr[:, : dim + 2], fene_r0], dim=-1)

    pair_index, pair_attr = pair_edges(
        graph, force_field, bonded_pairs=graph.edge_index if force_field.exclude_bonded else None
    )

    graph.edge_index = torch.cat([graph.edge_index, pair_index], dim=-1)
    graph.edge_attr = torch.cat([bond_attr, pair_attr], dim=0)
    return graph


def prepare_frames(
    frames: list[Data],
    force_field: KGPotential | None = None,
    *,
    pair_edges_on: str = "all",
) -> list[Data]:
    """Turn raw frames into prepared graphs: symmetrised, optionally with pair edges.

    `pair_edges_on="all"` is the honest default and what a rollout or an energy
    calculation needs. `"last"` builds the neighbour list on the final frame
    only, for a model input window where the earlier frames are read for their
    positions alone. The neighbour list is by far the largest part of a prepared
    frame -- about 21k edges against 660 half-bonds for a 315-bead melt at the
    full 2.5 sigma cutoff -- so this is the difference between a window costing
    one neighbour list and costing `history + 1` of them.

    Passing `force_field=None`, or a potential with no pair interaction such as a
    harmonic spring network, skips the pair edges entirely and leaves a
    symmetrised bond-only graph -- which is what the `edges="bond"` input-graph
    mode wants, and the only thing a spring network has.
    """
    if pair_edges_on not in ("all", "last"):
        raise ValueError(f"unknown pair_edges_on {pair_edges_on!r}; expected 'all' or 'last'.")

    prepared = []
    for index, frame in enumerate(frames):
        if not isinstance(getattr(frame, "box_tensor", None), Tensor):
            frame = add_box_tensor(frame)
        dim = infer_dim(frame)
        if frame.edge_attr.shape[1] == typed_width(dim):
            raise ValueError("`prepare_frames` expects raw frames; this one is already typed.")

        graph = to_undirected_graph(frame)
        wants_pairs = force_field is not None and force_field.has_pair_interaction
        if wants_pairs and (pair_edges_on == "all" or index == len(frames) - 1):
            graph = append_pair_interactions(graph, force_field)
        graph.pos = graph.x.clone()
        # The raw bond topology, carried along so that a predicted frame can be
        # rebuilt in the dataset's own schema. These three tensors are shared by
        # every frame of a trajectory, so carrying them costs nothing.
        graph.bond_index = frame.edge_index
        graph.bond_attr = frame.edge_attr
        graph.bond_types = frame.bond_types
        prepared.append(graph)

    return prepared
