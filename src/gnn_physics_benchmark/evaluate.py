"""Score a model on held-out trajectories; optional stress analysis is separate."""

from __future__ import annotations

from . import metrics
from .data import registry
from .graph.features import potential_for
from .training.rollout import rollout


def evaluate_model(model, trajectories, config, *, split: str = "test", steps: int | None = None, sample_stride: int = 5) -> dict:
    """Roll a model out on one split and summarise everything.

    Returns the rollout metrics -- raw position MSE at the last frame reached,
    the frozen baseline it is read against, and their ratio -- together with the
    Poisson's ratio R^2 along the rollout, and the stress response scores when
    `train.stress_metrics` is on.
    """
    entry = registry.get(config.dataset)
    potential = potential_for(config.graph, entry)
    graph_spec = config.graph
    steps = steps or config.train.val_rollout_steps

    errors, entries, rollouts = [], [], 0
    for trajectory in trajectories:
        available = len(trajectory) - graph_spec.window_length
        if available < 1:
            continue
        predicted = rollout(
            model,
            trajectory[: graph_spec.window_length],
            min(steps, available),
            graph_spec,
            potential,
            box_mode=config.train.box_mode,
            device=config.train.device,
        )
        errors.append(
            metrics.rollout_errors(
                predicted, trajectory, graph_spec.history, driven_axis=entry.driven_axis
            )
        )
        # Off unless asked for: they need the dataset's force field and cost a
        # virial per sampled frame.
        if config.train.stress_metrics and potential is not None:
            entries.append(
                metrics.stress_entry(
                    predicted,
                    trajectory,
                    potential,
                    history=graph_spec.history,
                    sample_stride=sample_stride,
                    device=config.train.device,
                )
            )
        rollouts += 1

    summary = {"split": split, "systems": rollouts, "requested_steps": steps}
    summary.update(metrics.summarise_rollouts(errors, steps))
    summary.update(metrics.summarise_stress(entries))
    summary["parameters"] = sum(p.numel() for p in model.parameters())
    return summary
