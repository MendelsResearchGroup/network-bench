"""Run a benchmark: prepare data, train, evaluate, save results."""

from __future__ import annotations

import traceback
import math
from dataclasses import replace
from pathlib import Path

import torch

from . import models
from .data import loading, registry
from .graph.features import build_input_graph, potential_for, prepare_window
from .interfaces.raw import validate_raw_frame
from .evaluate import evaluate_model
from .results import cache
from .training.config import RunConfig
from .training.loop import load_checkpoint, train
from .training.loss import fit_target_scale

__all__ = ["run", "select_epoch"]


def select_epoch(history: dict, key: str) -> dict:
    """The validation entry of `history` that scored best on `key`."""
    scored = [entry for entry in history["epochs"] if math.isfinite(entry.get(key, float("nan")))]
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
    entry = registry.get(config.dataset)
    config = replace(config, graph=config.graph.resolved(entry.schema))
    spec = config.split
    resolved = loading.resolve_split(
        entry, explicit=spec.explicit, manifest=spec.manifest, ratios=spec.ratios,
        sizes=spec.sizes, seed=spec.seed, limit=spec.limit,
    )
    directory = cache.run_dir(config, resolved, root)
    if resume and (directory / "metrics.json").is_file():
        if verbose:
            print(f"{directory.name}: already done, skipping")
        return cache.read_result(directory)

    try:
        data = {part: load_data(config, systems) for part, systems in resolved.items()}
        scale = fit_target_scale(
            data["train"], config.graph, config.train,
            max_windows=config.train.target_scale_windows, device=config.train.device,
        )
        torch.manual_seed(config.train.seed)
        model = models.build(config.model, config.graph, scale, **config.model_hyperparameters).to(config.train.device)
        potential = potential_for(config.graph, entry)
        window = data["train"][0][:config.graph.window_length]
        graph = build_input_graph(prepare_window(window, config.graph, potential), config.graph, potential)
        model.eval()
        with torch.no_grad():
            validate_raw_frame(model(graph.to(config.train.device)), entry.schema, name=f"{config.model} output")
        history = train(model, data, config, scale, save_dir=directory / "checkpoints", verbose=verbose)
        selection = config.train.selection_metric
        if selection is not None:
            selected = select_epoch(history, selection)
            state, _, _ = load_checkpoint(directory / "checkpoints" / f"epoch_{selected['epoch']:04d}.pt")
            model.load_state_dict(state)
        measured = evaluate_model(model, data[split], config, split=split)
        measured["trained_epochs"] = len(history["epochs"])
        measured["early_stopped"] = history["early_stopped"]
        if selection is not None:
            measured["selected_epoch"] = selected["epoch"]
            measured[f"val_{selection}"] = selected[selection]
        measured["target_sigma"] = scale.std().flatten().tolist()
    except Exception:
        cache.write_result(directory, config, resolved, traceback_text=traceback.format_exc())
        raise

    cache.write_result(directory, config, resolved, metrics=measured, history=history)
    return cache.read_result(directory)


def load_data(config: RunConfig, systems: list[str]) -> list[list]:
    """Load just the named systems using the run's frame selection."""
    spec = config.split
    return loading.load_trajectories(
        registry.get(config.dataset), systems, first_frame=spec.first_frame,
        frame_stride=spec.frame_stride, max_frames=spec.max_frames,
    )
