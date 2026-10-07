"""How well a model reproduces a trajectory.

The headline number is the position MSE at the last frame the rollout reached,
read against the baseline of predicting nothing at all: every bead frozen at the
last seed frame. `relative_mse` is the ratio of the two, so 1.0 means the model
did no better than standing still and below 1.0 means it did.

Poisson's ratio is scored along the rollout: `poisson_r2_<k>` is the R^2 across
systems of the ratio `k` steps past the last seed frame, every `POISSON_EVERY`
steps. It comes from the node positions, not the box: a rollout does not predict
the box, its transverse edges stay put, so a box-based ratio would be the same
for every model. On ground truth the position-based ratio matches the box-based
one to R^2 >= 0.99 on the inverse-design sets.

Position MSE, the frozen baseline and relative MSE are also recorded at these
horizons. Each position MSE averages squared errors over node coordinates first,
then averages those per-system errors across the networks that reached the step.

Relative MSE is a ratio of means over systems rather than a mean of ratios. A single system
whose frozen baseline happens to be tiny would otherwise dominate the average.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

__all__ = ["r2_centred", "position_poisson_ratio", "rollout_errors", "summarise_rollouts"]

#: Steps between the rollout horizons Poisson's ratio is scored at.
POISSON_EVERY = 10


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


def position_poisson_ratio(reference: Data, frame: Data, *, driven_axis: int = 0) -> float:
    """Poisson's ratio between two frames, from how the nodes moved.

    The transverse strain is the affine stretch that best explains the node
    displacements -- the least-squares slope of displacement against position --
    and the axial strain is the imposed one, read off the box. Displacements are
    taken to the nearest periodic image.
    """
    lengths = frame.box_tensor
    axial = float((lengths[driven_axis] - reference.box_tensor[driven_axis]) / reference.box_tensor[driven_axis])
    if abs(axial) < 1e-12:
        return float("nan")
    strains = []
    for axis in range(lengths.numel()):
        if axis == driven_axis:
            continue
        position = reference.pos[:, axis] - reference.pos[:, axis].mean()
        moved = frame.pos[:, axis] - reference.pos[:, axis]
        moved = moved - lengths[axis] * torch.round(moved / lengths[axis])
        moved = moved - moved.mean()
        strains.append(float((moved * position).sum() / (position * position).sum()))
    return -sum(strains) / len(strains) / axial


def box_poisson_ratio(reference: Data, frame: Data, *, driven_axis: int = 0) -> float:
    strain = (frame.box_tensor - reference.box_tensor) / reference.box_tensor
    if abs(float(strain[driven_axis])) < 1e-12:
        return float("nan")
    free = [axis for axis in range(len(strain)) if axis != driven_axis]
    return float(-strain[free].mean() / strain[driven_axis])


def rollout_errors(predicted: list[Data], truth: list[Data], history: int, *, driven_axis: int = 0,
                   poisson_method: str = "affine") -> dict:
    """Position errors of one rolled-out trajectory against the ground truth.

    `predicted` includes the seed frames, so it indexes like `truth`. The frozen
    baseline is the last seed frame, `predicted[history]`, compared against the
    same true frame the model's last step is compared against.
    """
    reached = len(predicted)
    reference = truth[:reached]
    last_true = reference[-1].pos
    seed = truth[history]
    # A clamped transverse box has no Poisson's ratio to score.
    clamped = all(
        float(reference[-1].box_tensor[axis]) == float(seed.box_tensor[axis])
        for axis in range(seed.box_tensor.numel()) if axis != driven_axis
    )
    nu, mse = {}, {}
    side_indices = None
    if poisson_method == "sides":
        from .sides import directional_side_indices_from_box
        side_indices = directional_side_indices_from_box(predicted[history])
    for step in range(POISSON_EVERY, reached - history, POISSON_EVERY):
        target = truth[history + step].pos
        mse[step] = (
            float(torch.nn.functional.mse_loss(predicted[history + step].pos, target)),
            float(torch.nn.functional.mse_loss(predicted[history].pos, target)),
        )
        if clamped:
            nu[step] = (float("nan"), float("nan"))
            continue
        if poisson_method == "affine":
            nu[step] = (
                position_poisson_ratio(seed, truth[history + step], driven_axis=driven_axis),
                position_poisson_ratio(predicted[history], predicted[history + step], driven_axis=driven_axis),
            )
        else:
            target_nu = box_poisson_ratio(seed, truth[history + step], driven_axis=driven_axis)
            if poisson_method == "box":
                estimate = box_poisson_ratio(predicted[history], predicted[history + step], driven_axis=driven_axis)
            else:
                from .sides import calc_p_ratio_rollout_sides
                estimate = calc_p_ratio_rollout_sides([predicted[history], predicted[history + step]],
                                                     side_idx=side_indices)
            nu[step] = (target_nu, estimate)
    return {
        "position_mse": float(torch.nn.functional.mse_loss(predicted[-1].pos, last_true)),
        "frozen_mse": float(torch.nn.functional.mse_loss(predicted[history].pos, last_true)),
        "steps_completed": reached - history - 1,
        "nu": nu,
        "mse": mse,
    }


def summarise_rollouts(errors: list[dict], requested_steps: int) -> dict:
    """Aggregate per-system rollout errors into the reported metrics."""
    if not errors:
        return {
            "position_mse": float("nan"),
            "frozen_mse": float("nan"),
            "relative_mse": float("nan"),
            "steps_completed": 0.0,
            "diverged": 0,
        }
    mean_mse = sum(error["position_mse"] for error in errors) / len(errors)
    mean_frozen = sum(error["frozen_mse"] for error in errors) / len(errors)
    summary = {
        "position_mse": mean_mse,
        "frozen_mse": mean_frozen,
        "relative_mse": mean_mse / mean_frozen if mean_frozen > 0 else float("nan"),
        "steps_completed": sum(error["steps_completed"] for error in errors) / len(errors),
        "diverged": sum(1 for error in errors if error["steps_completed"] < requested_steps),
    }
    # Each horizon over the systems whose rollout reached it.
    for step in range(POISSON_EVERY, requested_steps + 1, POISSON_EVERY):
        pairs = [error["nu"][step] for error in errors if step in error["nu"]]
        summary[f"poisson_r2_{step}"] = r2_centred([p[0] for p in pairs], [p[1] for p in pairs])
        mse = [error["mse"][step] for error in errors if step in error["mse"]]
        position = sum(pair[0] for pair in mse) / len(mse) if mse else float("nan")
        frozen = sum(pair[1] for pair in mse) / len(mse) if mse else float("nan")
        summary[f"position_mse_{step}"] = position
        summary[f"frozen_mse_{step}"] = frozen
        summary[f"relative_mse_{step}"] = position / frozen if frozen > 0 else float("nan")
    return summary
