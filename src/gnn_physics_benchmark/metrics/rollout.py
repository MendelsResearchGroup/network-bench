"""How well a model reproduces a trajectory.

The headline number is the position MSE at the last frame the rollout reached,
read against the baseline of predicting nothing at all: every bead frozen at the
last seed frame. `relative_mse` is the ratio of the two, so 1.0 means the model
did no better than standing still and below 1.0 means it did.

It is a ratio of means over systems rather than a mean of ratios. A single system
whose frozen baseline happens to be tiny would otherwise dominate the average.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

__all__ = ["r2_centred", "mean_poisson_ratio", "rollout_errors", "summarise_rollouts"]


def r2_centred(truth: list[float], predicted: list[float]) -> float:
    """Coefficient of determination against the spread of `truth`.

    Centred: the denominator is the variance of the true values across systems.
    This is the right convention for comparing one scalar per system -- a
    Poisson's ratio, a stress slope -- and it is *not* the convention used for
    the acceleration floors in `metrics.ladder`, which are uncentred because
    there the target's mean is physically zero.
    """
    values = [value for value in zip(truth, predicted) if all(v == v for v in value)]
    if len(values) < 2:
        return float("nan")
    truth, predicted = [list(column) for column in zip(*values)]
    mean = sum(truth) / len(truth)
    total = sum((value - mean) ** 2 for value in truth)
    residual = sum((value - other) ** 2 for value, other in zip(truth, predicted))
    return 1.0 - residual / total if total > 0 else float("nan")


def mean_poisson_ratio(trajectory: list[Data], *, driven_axis: int = 0) -> float:
    """Transverse strain over axial strain, between the first and last frame.

    Averaged over every axis that is not the driven one, so this is `nu_y` in two
    dimensions and `(nu_y + nu_z) / 2` in three. Read off the box alone -- no
    particle position enters.

    A dataset whose transverse box is clamped has a Poisson's ratio of identically
    zero by construction; the `fixyz` sets are why this is reported as NaN there
    rather than as a number, and why the stress-based `ratio_r2` exists.
    """
    first, last = trajectory[0], trajectory[-1]
    axes = [axis for axis in range(first.box_tensor.numel()) if axis != driven_axis]
    axial = (last.box_tensor[driven_axis] - first.box_tensor[driven_axis]) / first.box_tensor[driven_axis]
    if abs(float(axial)) < 1e-12:
        return float("nan")
    transverse = [
        (last.box_tensor[axis] - first.box_tensor[axis]) / first.box_tensor[axis] for axis in axes
    ]
    return float(-sum(transverse) / len(axes) / axial)


def rollout_errors(predicted: list[Data], truth: list[Data], history: int, *, driven_axis: int = 0) -> dict:
    """Position errors of one rolled-out trajectory against the ground truth.

    `predicted` includes the seed frames, so it indexes like `truth`. The frozen
    baseline is the last seed frame, `predicted[history]`, compared against the
    same true frame the model's last step is compared against.
    """
    reached = len(predicted)
    reference = truth[:reached]
    last_true = reference[-1].pos
    return {
        "position_mse": float(torch.nn.functional.mse_loss(predicted[-1].pos, last_true)),
        "frozen_mse": float(torch.nn.functional.mse_loss(predicted[history].pos, last_true)),
        "steps_completed": reached - history - 1,
        "predicted_nu": mean_poisson_ratio(predicted, driven_axis=driven_axis),
        "truth_nu": mean_poisson_ratio(reference, driven_axis=driven_axis),
    }


def summarise_rollouts(errors: list[dict], requested_steps: int) -> dict:
    """Aggregate per-system rollout errors into the reported metrics."""
    if not errors:
        return {
            "position_mse": float("nan"),
            "frozen_mse": float("nan"),
            "relative_mse": float("nan"),
            "poisson_r2": float("nan"),
            "steps_completed": 0.0,
            "diverged": 0,
        }
    mean_mse = sum(error["position_mse"] for error in errors) / len(errors)
    mean_frozen = sum(error["frozen_mse"] for error in errors) / len(errors)
    return {
        "position_mse": mean_mse,
        "frozen_mse": mean_frozen,
        "relative_mse": mean_mse / mean_frozen if mean_frozen > 0 else float("nan"),
        "poisson_r2": r2_centred(
            [error["truth_nu"] for error in errors], [error["predicted_nu"] for error in errors]
        ),
        "steps_completed": sum(error["steps_completed"] for error in errors) / len(errors),
        "diverged": sum(1 for error in errors if error["steps_completed"] < requested_steps),
    }
