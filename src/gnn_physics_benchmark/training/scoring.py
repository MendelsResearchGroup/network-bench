"""Shared rollout scoring for validation and held-out evaluation."""
from ..metrics.rollout import rollout_errors, summarise_rollouts
from .rollout import rollout


def score_rollouts(model, trajectories, graph_spec, potential, spec, *, steps,
                   driven_axis=0, barostat=None, poisson_method="affine", observe=None, requested_steps=None):
    errors = []
    for trajectory in trajectories:
        available = len(trajectory) - graph_spec.window_length
        if available < 1:
            continue
        predicted = rollout(model, trajectory[:graph_spec.window_length], min(steps, available),
                            graph_spec, potential, box_mode=spec.box_mode,
                            device=spec.device, barostat=barostat)
        errors.append(rollout_errors(predicted, trajectory, graph_spec.history,
                                    driven_axis=driven_axis, poisson_method=poisson_method))
        if observe is not None:
            observe(predicted, trajectory)
    return {"systems": len(errors), **summarise_rollouts(errors, steps if requested_steps is None else requested_steps)}
