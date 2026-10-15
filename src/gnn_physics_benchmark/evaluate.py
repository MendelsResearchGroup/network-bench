"""Score a model on held-out trajectories; optional stress analysis is separate."""

from __future__ import annotations

from .data import registry
from .graph.features import potential_for
from .training.scoring import score_rollouts


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
    barostat = None
    if config.barostat is not None:
        from .barostat import Barostat
        if not config.barostat:
            raise ValueError("Evaluation requires frozen barostat parameters; calibrate on training data first.")
        barostat = Barostat(entry, potential, config.barostat)
    steps = steps or config.train.val_rollout_steps

    entries = []
    observe = None
    if config.train.stress_metrics and potential is not None:
        from .analysis.stress import stress_entry, summarise_stress

        def observe(predicted, trajectory):
            entries.append(stress_entry(predicted, trajectory, potential,
                                       history=graph_spec.history, sample_stride=sample_stride,
                                       device=config.train.device))

    summary = {"split": split, "requested_steps": steps}
    summary.update(score_rollouts(model, trajectories, graph_spec, potential, config.train,
                                  steps=steps, driven_axis=entry.driven_axis, barostat=barostat,
                                  poisson_method=config.poisson_method, observe=observe))
    if observe is not None:
        summary.update(summarise_stress(entries))
    if graph_spec.prediction_stride != 1:
        summary["prediction_stride"] = graph_spec.prediction_stride
        summary["model_steps"] = (steps + graph_spec.prediction_stride - 1) // graph_spec.prediction_stride
    summary["parameters"] = sum(p.numel() for p in model.parameters())
    summary["barostat"] = config.barostat is not None
    summary["poisson_method"] = config.poisson_method
    return summary
