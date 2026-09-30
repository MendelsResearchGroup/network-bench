"""Running a benchmark: one model's full metric set.

`evaluate_model` rolls a trained model out on a split and returns everything the
results table reports. `run` ties a `RunConfig` to the cache: train, evaluate,
write, and skip a cell that is already done.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from . import metrics
from .data import registry
from .results import cache
from .training import RunConfig, build, rollout, train
from .training.loop import load_checkpoint

__all__ = ["evaluate_model", "select_epoch", "run"]


def evaluate_model(prepared, *, split: str = "test", steps: int | None = None, sample_stride: int = 5) -> dict:
    """Roll `prepared.model` out on one split and summarise everything.

    Returns the rollout metrics -- raw position MSE at the last frame reached,
    the frozen baseline it is read against, and their ratio -- together with the
    Poisson's ratio R^2 along the rollout, and the stress response scores when
    `train.stress_metrics` is on.
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
        # Off unless asked for: they need the dataset's force field and cost a
        # virial per sampled frame.
        if config.train.stress_metrics and prepared.potential is not None:
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


def select_epoch(history: dict, key: str) -> dict:
    """The validation entry of `history` that scored best on `key`."""
    scored = [entry for entry in history["epochs"] if entry.get(key, float("nan")) == entry.get(key)]
    lower = "loss" in key or "mse" in key
    return (min if lower else max)(scored, key=lambda entry: entry[key])


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
        if config.train.select_by is not None:
            selected = select_epoch(history, config.train.select_by)
            state, _, _ = load_checkpoint(directory / "checkpoints" / f"epoch_{selected['epoch']:04d}.pt")
            prepared.model.load_state_dict(state)
        measured = evaluate_model(prepared, split=split)
        if config.train.select_by is not None:
            measured["selected_epoch"] = selected["epoch"]
            measured[f"val_{config.train.select_by}"] = selected[config.train.select_by]
        measured["target_sigma"] = prepared.scale.std().flatten().tolist()
    except Exception:
        cache.write_result(directory, config, resolved, traceback_text=traceback.format_exc())
        raise

    cache.write_result(directory, config, resolved, metrics=measured, history=history)
    return cache.read_result(directory)
