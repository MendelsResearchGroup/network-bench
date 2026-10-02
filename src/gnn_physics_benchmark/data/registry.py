"""The dataset registry: a user-chosen key for every set of trajectories.

A dataset key is the identity a result is cached under, so it has to pin down
everything about the data that a number could depend on: which directory, which
systems are usable, how often frames were dumped, and which force field generated
them. Adding a dataset means adding one `DatasetEntry` here and dropping the
trajectories under the data root.

The data itself is never in the repository. `resolve_root` looks under
`$GNN_BENCH_DATA_ROOT` when that is set and otherwise under `~/work/data`,
so installed packages and cluster jobs use the same shared data directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..interfaces.raw import FrameSchema

__all__ = ["DatasetEntry", "DATASETS", "resolve_root", "get", "keys"]


def resolve_root() -> Path:
    """Where dataset directories live."""
    override = os.environ.get("GNN_BENCH_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "work" / "data"


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset.

    `directory` is relative to the data root. `description` is prose for a human
    reading a results table; `field_notes` documents anything about *this* set
    that the generic raw schema cannot say, such as which box axes are free.
    """

    key: str
    directory: str
    description: str
    schema: FrameSchema
    """What this dataset's stored frames hold: the edge columns and so the spatial
    dimension, which fields are stored rather than derived, and the module its
    `Box` was pickled from."""
    interval: int
    """MD steps between consecutive dumped frames."""
    frames: int
    """Frames per trajectory as generated."""
    driven_axis: int
    """The box axis compression is applied to (0 = x)."""
    free_axes: tuple[int, ...]
    """Box axes allowed to respond. Empty means the transverse box is clamped, in
    which case Poisson's ratio is not measurable from the trajectory."""
    potential: dict | None = None
    """How to evaluate the force field of this dataset, as
    `{"kind": ..., **constants}`; `None` for a dataset whose force field is not
    modelled, which then supports bond-only graphs and no stress metrics. The
    constants are a property of the data, not of a run."""
    manifests: dict[str, str] = field(default_factory=dict)
    """Named lists of usable systems, e.g. "clean" -> "clean_1024.json". A run
    selects one by name; the resulting membership list is hashed into the result
    key so that regenerating a manifest misses the cache instead of silently
    comparing unlike splits."""
    field_notes: str = ""

    def root(self) -> Path:
        return resolve_root() / self.directory

    def manifest(self, name: str) -> list[str]:
        """The system stems named by one manifest, in file order."""
        if name not in self.manifests:
            raise KeyError(f"dataset {self.key!r} has no manifest {name!r}; known: {sorted(self.manifests)}.")
        path = self.root() / self.manifests[name]
        if not path.is_file():
            raise FileNotFoundError(f"manifest {name!r} of dataset {self.key!r} is not at {path}.")
        return json.loads(path.read_text())

    def systems(self) -> list[str]:
        """Every system stem present on disk, sorted."""
        return sorted(path.stem for path in self.root().glob("*.pt"))


#: The GNNInverseDesign spring networks: `node_optimized`, `stiff_optimized` and
#: `noisy_dump200` store one layout, and it is the same four columns as
#: `dePablo_random`. What differs between them is what the stiffness column
#: *holds*, which is why every entry declares it read from the column and never
#: reconstructs it from the rest length -- see each entry's notes.
_INVERSE_DESIGN_SCHEMA = FrameSchema(
    edge_columns=("dx", "dy", "r", "stiffness", "rest_length"),
    derived_columns=("rest_length",),
    stored_fields=(),
    pickle_module="network",
    column_docs={
        "rest_length": (
            "the bond's length in the first stored frame, the undeformed reference state "
            "after LAMMPS' box/relax minimisation. Computed at load time, not stored"
        ),
        "stiffness": (
            "per-bond spring constant: the LAMMPS `bond_coeff` K, constant in time. How it "
            "relates to the rest length differs by dataset (1/l0, 1/l0 with noise, or "
            "optimised and unrelated), so it is always read from this column"
        ),
    },
)

#: The LAMMPS decks behind the GNNInverseDesign sets: `fix langevin` + `fix nph`
#: on y, `fix deform` on x at erate 1e-5 with dt 0.01, dumped every 200 steps.
_INVERSE_DESIGN_POTENTIAL = {
    "kind": "harmonic",
    "stiffness_from": "column",
    "stiffness_column": "stiffness",
    "rest_length_from": "first_frame",
}

_INVERSE_DESIGN_NOTES = (
    " LAMMPS' `bond_style harmonic` is E = K (r - l0)^2 with no 1/2, so its bond force "
    "is 2K(r - l0); the benchmark's harmonic potential uses k(r - l0). Every absolute "
    "force and stress here is therefore half of LAMMPS'. Nothing scored depends on it: "
    "ground-truth and predicted stress go through the same virial, and R^2 and "
    "relative MSE are scale free. Positions are float32 and there is no thermo log."
)


DATASETS: dict[str, DatasetEntry] = {
    "noisy_lj": DatasetEntry(
        key="noisy_lj",
        directory="noisy-lj",
        description="200 noisy spring-network trajectories with LJ interactions, 1500 stored frames each.",
        schema=FrameSchema(
            edge_columns=("dx", "dy", "r", "stiffness"),
            stored_fields=(),
            pickle_module="network",
        ),
        interval=1,
        frames=1500,
        driven_axis=0,
        free_axes=(1,),
        field_notes=(
            "Source: data/noisy-lj/chunk_*.pt. Time is counted in stored frames; "
            "the files do not record the MD dump interval. Benchmark inputs use "
            "the stored bond graph only. No rest lengths or LJ force parameters "
            "are inferred, and force/stress metrics are unavailable."
        ),
    ),
    "cold_wca_fixyz_n1024": DatasetEntry(
        key="cold_wca_fixyz_n1024",
        directory="graphs_cold_m1_d10_wca_fixyz_n1024_i200_inst",
        description=(
            "1024 Kremer-Grest polymer melts compressed uniaxially along x at a strain "
            "rate of 1e-5 per timestep, essentially at zero temperature (Langevin damping "
            "10, peak temperatures around 1e-5 in LJ units). The pair interaction is "
            "purely repulsive WCA and the transverse box is held fixed. Positions are "
            "instantaneous dumps, not time-averaged, unwrapped and shifted into the first "
            "image. 400 frames per system, 200 MD steps apart. Inertial dynamics."
        ),
        schema=FrameSchema(
            edge_columns=("dx", "dy", "dz", "r", "fene_k"),
            stored_fields=("pos", "box_tensor", "time", "atom_ids", "atom_types", "molecule_ids", "bond_types"),
            pickle_module="network_minimal",
            column_docs={
                "fene_k": "FENE spring constant, constant 30.0 across every bond",
            },
        ),
        interval=200,
        frames=400,
        driven_axis=0,
        free_axes=(),
        potential={
            "kind": "kremer_grest",
            "epsilon": 1.0,
            "sigma": 1.0,
            "cutoff": 1.122462048309373,  # WCA: the LJ minimum, shifted and truncated
            "fene_k": 30.0,
            "fene_r0": 1.5,
        },
        manifests={
            # Despite the shared prefix these two say different things.
            # `clean_1024.json` is simply the roster of all 1024 generated
            # systems; `clean_400.json` (identical to `clean.json`) is the 771
            # of them whose potential energy shows no plastic drop over the 400
            # dumped frames, i.e. the actual cleanliness filter.
            "all": "clean_1024.json",
            "clean": "clean_400.json",
        },
        field_notes=(
            "Ly and Lz are constant by construction, so the transverse strain is "
            "identically zero and the box-based Poisson's ratio is not measurable "
            "here: the `poisson_r2_<k>` metrics report NaN. The mechanical response "
            "still carries it, though -- at clamped transverse box the ratio of "
            "the transverse to the axial stress-strain slope is nu/(1-nu), which "
            "gives nu = 0.445 on rubber_0000, matching the LAMMPS log. That is "
            "what the `ratio_r2` stress metric scores. Systems outside the `clean` "
            "manifest show plastic rearrangements (potential-energy drops) inside "
            "the frames training reads; they are not hard examples but unlearnable "
            "ones, and they also inflate the target normaliser for every other "
            "system in the set, so exclude them unless you are studying plasticity "
            "itself. `logs/<system>.log` holds the LAMMPS thermo output, including "
            "the pressure tensor used as ground truth for the stress metrics."
        ),
    ),
    "dePablo_random": DatasetEntry(
        key="dePablo_random",
        directory="dePablo_random",
        description=(
            "301 two-dimensional random spring networks compressed along x at constant "
            "rate, with the transverse box free to respond. Roughly 100 nodes and 300 "
            "harmonic bonds each, 1500 frames per system. Every bond has its own "
            "stiffness and its own rest length. The dynamics are overdamped: the net "
            "force points along the next displacement rather than along the acceleration."
        ),
        schema=FrameSchema(
            edge_columns=("dx", "dy", "r", "stiffness", "rest_length"),
            derived_columns=("rest_length",),
            stored_fields=(),
            pickle_module="network",
            column_docs={
                "rest_length": (
                    "the bond's length in the first stored frame, which is the undeformed "
                    "reference state of these networks. Computed at load time, not stored"
                ),
                "stiffness": (
                    "per-bond spring constant, used directly as the coefficient of "
                    "(r - l0) in the bond force. In this dataset it happens to equal "
                    "1/l0**2; that is a property of how this set was generated, not a "
                    "rule -- other harmonic sets define the stiffness as 1/l0 instead, "
                    "which is why the potential declares where it comes from"
                ),
            },
        ),
        interval=1,
        frames=1500,
        driven_axis=0,
        free_axes=(1,),
        potential={
            "kind": "harmonic",
            # The stiffness is read straight from the named edge column. The
            # alternative, for a set that does not store it, is
            # "stiffness_from": "inverse_rest_length", i.e. k = 1 / l0.
            "stiffness_from": "column",
            "stiffness_column": "stiffness",
            "rest_length_from": "first_frame",
        },
        manifests={},
        field_notes=(
            "Ly responds while Lx is driven, so Poisson's ratio IS measurable here and "
            "it varies widely between systems -- 0.003, 0.095, 0.282, 0.285 on the four "
            "systems sampled -- which makes it a discriminating target rather than the "
            "near-constant it is on the Kremer-Grest sets. The rest length of every bond "
            "is its length in frame 0, the undeformed reference state; a consequence is "
            "that a mechanical-equilibrium check at frame 0 is vacuous, satisfied by any "
            "stiffness at all. Positions are stored in float32 at order 5 sigma while the "
            "one-step target is order 6e-7, so the quantisation ceiling bites: r2_ceiling "
            "runs 0.78 to 0.94 here against 0.997 on the Kremer-Grest set. There is no "
            "pair interaction and no thermo log."
        ),
    ),
    "node_optimized": DatasetEntry(
        key="node_optimized",
        directory="node_optimized",
        description=(
            "381 two-dimensional spring networks whose node positions were optimised for a "
            "target Poisson's ratio, compressed along x to 1% strain with the transverse box "
            "barostatted to zero stress. 100-196 nodes (median 148), about 2.9 bonds per "
            "node, 500 frames per system 200 MD steps apart, 2e-5 strain per frame. Poisson's "
            "ratio -0.45 to 0.35, median -0.03. Overdamped: the net force points along the "
            "next displacement rather than along the acceleration. These are the "
            "`data_mini.tar.gz` subset of the Zenodo record (doi:10.5281/zenodo.20181262), "
            "whose full `data.tar.gz` holds 1998 such systems."
        ),
        schema=_INVERSE_DESIGN_SCHEMA,
        interval=200,
        frames=500,
        driven_axis=0,
        free_axes=(1,),
        potential=_INVERSE_DESIGN_POTENTIAL,
        field_notes=(
            "The stiffness column is exactly 1/l0 on every bond (k * l0 = 1 to float32 "
            "precision), so reading the column and `inverse_rest_length` coincide here; "
            "the column is declared anyway so all three inverse-design sets are read the "
            "same way. Shorter than the other two: 500 frames, 1% total strain."
            + _INVERSE_DESIGN_NOTES
        ),
    ),
    "stiff_optimized": DatasetEntry(
        key="stiff_optimized",
        directory="stiff_optimized",
        description=(
            "288 two-dimensional spring networks whose per-bond stiffnesses were optimised "
            "for a target Poisson's ratio, compressed along x to 3% strain with the "
            "transverse box barostatted to zero stress. 100-400 nodes (median 240), about "
            "2.9 bonds per node, 1500 frames per system 200 MD steps apart, 2e-5 strain per "
            "frame. Two families: 242 `chunk_<n>` (Poisson's ratio median -0.10, down to "
            "-0.77) and 46 `chunk_highP_<n>` (median +0.16). Overdamped. These are the "
            "`data_mini.tar.gz` subset of the Zenodo record (doi:10.5281/zenodo.20181262), "
            "whose full `data.tar.gz` holds 1085 such systems (930 plain, 155 highP)."
        ),
        schema=_INVERSE_DESIGN_SCHEMA,
        interval=200,
        frames=1500,
        driven_axis=0,
        free_axes=(1,),
        potential=_INVERSE_DESIGN_POTENTIAL,
        field_notes=(
            "The per-bond stiffnesses are the optimisation variable, so they bear no "
            "relation to the rest length any more: k runs from 3e-6 to 1.5. The optimiser "
            "keeps each bond between cut and its 1/l0 value -- k * l0 lies strictly inside "
            "(0, 1), with about 8% of bonds below 0.01 in the full set, i.e. all but cut. Not "
            "even the unoptimised members follow 1/l0: 25 systems (22 plain, 3 highP) are the "
            "optimiser's starting point, k = 0.5 / l0 exactly, Poisson's ratio 0.33-0.38, and "
            "they are kept in the set. "
            "Reconstructing the stiffness as 1/l0 is wrong here, not approximate. The "
            "dynamics are overdamped, so the non-affine displacement should follow the net "
            "force: early in the compression the column's bond forces explain about 0.97 of "
            "it and 1/l0 0.0-0.5, no better than giving every bond the same stiffness. The "
            "fit weakens further along a trajectory for any stiffness -- over the whole "
            "run the column gives 0.68-0.91 on sampled systems against 0.27-0.48. "
            "Anything that uses the force field (the `node_force` feature, the stress "
            "metrics) must read the column, which is what the potential declares."
            + _INVERSE_DESIGN_NOTES
        ),
    ),
    "noisy_dump200": DatasetEntry(
        key="noisy_dump200",
        directory="noisy_dump200",
        description=(
            "401 two-dimensional spring networks with noisy per-bond stiffnesses, compressed "
            "along x to 3% strain with the transverse box barostatted to zero stress. 82-113 "
            "nodes (median 96), about 2.0 bonds per node, 1500 frames per system 200 MD "
            "steps apart, 2e-5 strain per frame. Poisson's ratio -0.80 to 0.95, median "
            "0.14 -- the widest spread of the three inverse-design sets. Overdamped."
        ),
        schema=_INVERSE_DESIGN_SCHEMA,
        interval=200,
        frames=1500,
        driven_axis=0,
        free_axes=(1,),
        potential=_INVERSE_DESIGN_POTENTIAL,
        field_notes=(
            "The stiffness is 1/l0 with noise on part of the bonds: about 55% of them "
            "(47-63% per system) sit exactly on 1/l0 and the rest are scattered over "
            "k * l0 = 0.45-1.6, for 1.01 +- 0.13 overall. Like `stiff_optimized` it must be "
            "read from the column. The noise is what "
            "sets the mechanics: on sampled trajectories the column's bond forces explain "
            "0.45-0.97 of the non-affine displacement and 1/l0 explains 0.01-0.10, worse "
            "than uniform springs. Sparser than the other sets, about two bonds per node."
            + _INVERSE_DESIGN_NOTES
        ),
    ),
}


def keys() -> list[str]:
    return sorted(DATASETS)


def get(key: str) -> DatasetEntry:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset key {key!r}; registered: {keys()}.")
    return DATASETS[key]
