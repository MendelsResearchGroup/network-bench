"""Running a benchmark: dataset difficulty, and one model's full metric set.

`measure_difficulty` describes a dataset before any model touches it.
`evaluate_model` rolls a trained model out on a split and returns everything the
results table reports. `run` ties a `RunConfig` to the cache: train, evaluate,
write, and skip a cell that is already done.
"""

from __future__ import annotations

import statistics
import traceback
from pathlib import Path

import torch

from . import metrics
from .data import loading, registry
from .results import cache
from .training import RunConfig, build, rollout, train

__all__ = ["measure_difficulty", "evaluate_model", "run"]


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
        row = metrics.describe(*metrics.trajectory_arrays(trajectory), history=history, count=count)
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


def evaluate_model(prepared, *, split: str = "test", steps: int | None = None, sample_stride: int = 5) -> dict:
    """Roll `prepared.model` out on one split and summarise everything.

    Returns the rollout metrics -- raw position MSE at the last frame reached,
    the frozen baseline it is read against, and their ratio -- together with the
    Poisson's ratio R^2 and the stress response scores.
    """
    config = prepared.config
    graph_spec = config.graph
    steps = steps or config.train.val_rollout_steps
    trajectories = prepared.data[split]

    errors, entries, rollouts = [], [], 0
    for trajectory in trajectories:
        available = len(trajectory) - graph_spec.window_length
        if available < 1:
            continue
        predicted = rollout(
            prepared.model,
            trajectory[: graph_spec.window_length],
            min(steps, available),
            graph_spec,
            prepared.potential,
            box_mode=config.train.box_mode,
            device=config.train.device,
        )
        errors.append(
            metrics.rollout_errors(
                predicted, trajectory, graph_spec.history, driven_axis=prepared.entry.driven_axis
            )
        )
        # The stress metrics need a force field to sum a virial over. A dataset
        # that declares none, or a run on a bond-only graph, simply omits them.
        if prepared.potential is not None:
            entries.append(
                metrics.stress_entry(
                    predicted,
                    trajectory,
                    prepared.potential,
                    history=graph_spec.history,
                    sample_stride=sample_stride,
                    device=config.train.device,
                )
            )
        rollouts += 1

    summary = {"split": split, "systems": rollouts, "requested_steps": steps}
    summary.update(metrics.summarise_rollouts(errors, steps))
    summary.update(metrics.summarise_stress(entries))
    summary["parameters"] = sum(p.numel() for p in prepared.model.parameters())
    return summary


def run(
    config: RunConfig,
    *,
    root: str | Path | None = None,
    resume: bool = True,
    split: str = "test",
    verbose: bool = True,
) -> dict:
    """Train, evaluate and cache one benchmark cell.

    With `resume`, a cell whose `metrics.json` already exists is skipped and read
    back instead. A failure writes `failed.txt` and no metrics, so the next run
    retries it rather than treating it as done.
    """
    from .training.runner import resolve_split

    config = config.replace(graph=config.graph.resolved(registry.get(config.dataset).schema))
    resolved = resolve_split(config)
    directory = cache.run_dir(config, resolved, root)
    if resume and cache.run_exists(config, resolved, root):
        if verbose:
            print(f"{directory.name}: already done, skipping")
        return cache.read_result(directory)

    try:
        prepared = build(config)
        if verbose:
            print(prepared.summary(), flush=True)
        history = train(
            prepared.model,
            prepared.data,
            config,
            prepared.scale,
            save_dir=directory / "checkpoints",
            verbose=verbose,
        )
        measured = evaluate_model(prepared, split=split)
        measured["target_sigma"] = prepared.scale.std().flatten().tolist()
    except Exception:
        cache.write_result(directory, config, resolved, traceback_text=traceback.format_exc())
        raise

    cache.write_result(directory, config, resolved, metrics=measured, history=history)
    return cache.read_result(directory)
