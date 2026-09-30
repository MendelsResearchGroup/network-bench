"""Dataset difficulty: how hard a dataset is before any model touches it.

Kept apart from training and evaluation, which never import it. `gnn-bench
difficulty <dataset>` runs `measure_difficulty` and caches the result as
`results/<dataset>/difficulty.json`, which `notebooks/dataset_metrics.ipynb` reads.
"""

from __future__ import annotations

import statistics

from ..data import loading, registry
from .ladder import Samples, best_linear, linear_ladder, position_errors, r2_uncentred
from .measures import (
    alpha_from_positions,
    describe,
    floor_over_time,
    persistence_from_positions,
    quantisation_noise,
    samples_from_positions,
    trajectory_arrays,
)

__all__ = [
    "measure_difficulty",
    "Samples",
    "linear_ladder",
    "best_linear",
    "position_errors",
    "r2_uncentred",
    "describe",
    "floor_over_time",
    "trajectory_arrays",
    "samples_from_positions",
    "persistence_from_positions",
    "alpha_from_positions",
    "quantisation_noise",
]


def measure_difficulty(
    dataset: str,
    *,
    systems: int | None = 20,
    manifest: str | None = "clean",
    history: int = 3,
    count: int = 15,
    max_frames: int | None = None,
) -> dict:
    """Per-system difficulty rows for a dataset, plus their medians.

    Reads positions and boxes only, so no force field and no graph is built.

    `systems` are spread evenly over the sorted stems rather than taken from the
    front: stems follow generation order, the ensembles drift along it, and a
    lexicographic sort of `chunk_<n>` would bunch the first few into one stretch.
    """
    entry = registry.get(dataset)
    stems = sorted(entry.manifest(manifest)) if manifest else entry.systems()
    present = set(entry.systems())
    stems = [stem for stem in stems if stem in present]
    if systems and systems < len(stems):
        stems = stems[:: len(stems) // systems][:systems]

    rows = []
    for stem in stems:
        trajectory = loading.load_trajectory(entry, stem, max_frames=max_frames)
        row = describe(*trajectory_arrays(trajectory), history=history, count=count)
        rows.append({"system": stem, **row})

    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    median = {key: statistics.median(row[key] for row in rows) for key in numeric}
    return {
        "dataset": dataset,
        "manifest": manifest,
        "systems": len(rows),
        "history": history,
        "count": count,
        "median": median,
        "rows": rows,
    }
