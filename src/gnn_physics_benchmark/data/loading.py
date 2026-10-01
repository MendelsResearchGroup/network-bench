"""Reading trajectories off disk, and turning a split description into names.

Two things happen here that are worth knowing about.

First, the stored frames pickle a `Box` instance from whichever module the
generating code lived in -- `network_minimal` for the Kremer-Grest sets,
`network` for the older two-dimensional ones. Rather than ask every user to keep
those files on their path, `_install_pickle_alias` points the name the dataset
declares at `gnn_physics_benchmark.interfaces.box`, so `torch.load` resolves it.
`weights_only=False` is unavoidable for the same reason, which does mean loading
a trajectory executes pickle from the dataset directory.

Second, datasets differ in what they store. A frame is normalised on load against
the dataset's `FrameSchema`: anything in `DERIVABLE_FIELDS` that this dataset does
not store is filled in -- `pos` from `x`, `box_tensor` from `box`, `time` from the
frame index and the dump interval, and the identity fields from `arange` and
`ones`. Downstream code can then assume a complete frame without any dataset
knowing about any other.

Second, the train/validation/test split belongs to the *run*, not to the dataset.
The benchmark never supplies a default split silently: a run config states either
the system names outright or the manifest, ratios and seed that generate them, and
the resulting membership lists are hashed into the result key. Two runs that used
different systems can then never land in the same cache entry.
"""

from __future__ import annotations

import sys
import types

import torch
from torch_geometric.data import Data

from ..interfaces.box import Box
from ..interfaces.raw import FrameSchema, validate_raw_frame
from .registry import DatasetEntry

__all__ = [
    "load_trajectory",
    "load_trajectories",
    "normalize_frame",
    "resolve_split",
    "validate_dataset",
]


def _install_pickle_alias(module: str) -> None:
    """Make `<module>.Box` resolvable, so stored trajectories unpickle."""
    if module in sys.modules:
        return
    shim = types.ModuleType(module)
    shim.Box = Box
    shim.__doc__ = "Compatibility alias for gnn_physics_benchmark.interfaces.box."
    sys.modules[module] = shim


def _derive_edge_columns(frames: list[Data], schema: FrameSchema) -> None:
    """Append the edge columns this dataset computes rather than stores.

    Done on the untouched trajectory, before any `first_frame` or `frame_stride`
    slicing, because `rest_length` is read off the undeformed *stored* frame 0 --
    a run that starts part way in must still get the real rest lengths.
    """
    if not schema.derived_columns:
        return
    extra = []
    for name in schema.derived_columns:
        if name == "rest_length":
            extra.append(frames[0].edge_attr[:, schema.length_index : schema.length_index + 1].clone())
        else:  # pragma: no cover - guarded by FrameSchema
            raise ValueError(f"no rule for derived edge column {name!r}.")
    block = torch.cat(extra, dim=1)
    for frame in frames:
        frame.edge_attr = torch.cat([frame.edge_attr, block], dim=1)


def normalize_frame(frame: Data, schema: FrameSchema, *, index: int, interval: int) -> Data:
    """Fill in whatever this dataset does not store, so a frame is complete.

    Mutates and returns the freshly unpickled frame; nothing else holds it yet.
    The derived values are the only sensible ones -- `pos` is a clone of `x`
    because on a raw frame they are the same positions, `box_tensor` reads the
    pickled `Box`, and the identity fields are constant, so a dataset that does
    not distinguish particle types gets one type.
    """
    dim = schema.dim
    if not schema.stores("pos"):
        frame.pos = frame.x.clone()
    if not schema.stores("box_tensor"):
        lengths = frame.box.lengths[:dim]
        frame.box_tensor = torch.tensor(lengths, dtype=frame.x.dtype)
    if not schema.stores("time"):
        frame.time = index * interval
    num_nodes, num_edges = frame.x.shape[0], frame.edge_index.shape[1]
    if not schema.stores("atom_ids"):
        frame.atom_ids = torch.arange(1, num_nodes + 1)
    if not schema.stores("atom_types"):
        frame.atom_types = torch.ones(num_nodes, dtype=torch.long)
    if not schema.stores("molecule_ids"):
        frame.molecule_ids = torch.ones(num_nodes, dtype=torch.long)
    if not schema.stores("bond_types"):
        frame.bond_types = torch.ones(num_edges, dtype=torch.long)
    return frame


def load_trajectory(
    entry: DatasetEntry,
    stem: str,
    *,
    first_frame: int = 0,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[Data]:
    """One system's trajectory as a list of frames, in time order.

    `first_frame` and `frame_stride` are applied before `max_frames`, so the
    truncation counts kept frames. `frame_stride=k` is a k-times-coarser dump of
    the same trajectory, which changes what one step means and therefore what the
    model is asked to predict; it is part of the run config, not a loading detail.
    """
    if first_frame < 0 or frame_stride < 1:
        raise ValueError(f"need first_frame >= 0 and frame_stride >= 1, got {first_frame}, {frame_stride}.")
    _install_pickle_alias(entry.schema.pickle_module)
    path = entry.root() / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"system {stem!r} of dataset {entry.key!r} is not at {path}.")
    stored = torch.load(path, weights_only=False)
    _derive_edge_columns(stored, entry.schema)
    frames = stored[first_frame::frame_stride][:max_frames]
    if not frames:
        raise ValueError(f"{path} yielded no frames under first_frame={first_frame}, frame_stride={frame_stride}.")
    # A derived `time` counts from the start of the *stored* trajectory, so that
    # `first_frame` and `frame_stride` land on the simulation's own clock and the
    # gap between kept frames is `interval * frame_stride`.
    return [
        normalize_frame(
            frame, entry.schema, index=first_frame + index * frame_stride, interval=entry.interval
        )
        for index, frame in enumerate(frames)
    ]


def load_trajectories(
    entry: DatasetEntry,
    stems: list[str],
    *,
    first_frame: int = 0,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[list[Data]]:
    return [
        load_trajectory(
            entry, stem, first_frame=first_frame, frame_stride=frame_stride, max_frames=max_frames
        )
        for stem in stems
    ]


def resolve_split(
    entry: DatasetEntry,
    *,
    explicit: dict[str, list[str]] | None = None,
    manifest: str | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    sizes: tuple[int, int, int] | None = None,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, list[str]]:
    """Turn a split description into concrete `{"train"|"val"|"test": [stem, ...]}`.

    `explicit` names the systems of each part outright and is used as given. That
    is the reproducible option and the one a published comparison should use.

    Otherwise the systems of `manifest` (or everything on disk, if no manifest is
    named) are shuffled under `seed` and cut by `ratios`. The ensemble varies by
    random seed only, so there is nothing to stratify on and a shuffled split is
    the honest one. `sizes` gives the three parts as counts instead of ratios.
    `limit` keeps only the first `limit` systems *before* the cut,
    for quick smoke runs.
    """
    if explicit is not None:
        available = set(entry.systems())
        missing = [stem for stems in explicit.values() for stem in stems if stem not in available]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} system(s) of the explicit split are not under {entry.root()}, "
                f"e.g. {missing[:3]}."
            )
        return {name: list(stems) for name, stems in explicit.items()}

    stems = entry.manifest(manifest) if manifest is not None else entry.systems()
    present = set(entry.systems())
    stems = [stem for stem in sorted(stems) if stem in present]
    if not stems:
        raise FileNotFoundError(f"no systems of dataset {entry.key!r} are under {entry.root()}.")
    if limit is not None:
        stems = stems[:limit]

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios)}.")
    order = torch.randperm(len(stems), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [stems[index] for index in order]
    if sizes is not None:
        train_end, val_end = sizes[0], sizes[0] + sizes[1]
        test_end = val_end + sizes[2]
    else:
        train_end = int(len(shuffled) * ratios[0])
        val_end = train_end + int(len(shuffled) * ratios[1])
        test_end = len(shuffled)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:test_end],
    }


def validate_dataset(entry: DatasetEntry, *, systems: int = 1, frames: int | None = None) -> None:
    """Check that a registered dataset really matches the raw schema.

    Run this once when adding a dataset. It is not run during training.
    """
    stems = entry.systems()
    if not stems:
        raise FileNotFoundError(f"dataset {entry.key!r} has no .pt files under {entry.root()}.")
    for stem in stems[:systems]:
        trajectory = load_trajectory(entry, stem)
        for index, frame in enumerate(trajectory[:frames]):
            validate_raw_frame(frame, entry.schema, name=f"{entry.key}/{stem}[{index}]")
