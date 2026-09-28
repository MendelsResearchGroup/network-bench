"""How well increasingly informed *linear* predictors do on the target.

This is the floor a learned model has to clear. Every rung is fitted by least
squares on the same data it is scored on, so each is an upper bound on what that
family of linear predictors could achieve -- which is what a floor should be.

Which rung a model is held to matters, and getting it wrong reverses conclusions.
The rungs split into two families:

* `affine`, `kinematic` and `force` include the **affine column**,
  `x_t (L_{t+1}/L_t - 1) - x_{t-1} (L_t/L_{t-1} - 1)`. That column is built from
  the box at the *predicted* frame and scales with each bead's absolute position.
  No model here is given either, so these rungs describe the data rather than
  setting a bar. The reference project compared its models against `kinematic`
  for a long time, which made a graph network look worse than linear when it was
  in fact slightly better.
* `velocity` and `velocity_force` drop it. They see past residual velocities and
  nothing about the box -- exactly what the encoders are handed. **`velocity` is
  the floor to hold a model against.**

R^2 here is *uncentred*: `1 - mse / mean(target^2)`, the denominator being the
second moment rather than the variance. The acceleration's mean is physically
zero, so that is the honest normalisation, and it makes the rungs directly
comparable with a loss on a z-scored target. It is deliberately not the centred
convention used for per-system scalars in `metrics.rollout`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Samples", "r2_uncentred", "best_linear", "linear_ladder", "position_errors"]


@dataclass
class Samples:
    """One-step examples, flattened over beads, systems and frames.

    Every field is `(bead-steps, dim)` and lines up row for row.
    """

    target: Tensor
    """`(x_{t+1} - x_t) - (x_t - x_{t-1})`, the one-step acceleration."""
    affine: Tensor
    """The part of the target the box motion alone accounts for."""
    velocity: Tensor
    """Residual velocity into frame t."""
    velocity_prev: Tensor
    """Residual velocity into frame t-1."""
    force: Tensor
    """Conservative pair and bond force at frame t; zeros when not supplied."""


def r2_uncentred(target: Tensor, predicted: Tensor) -> float:
    return float(1.0 - (target - predicted).pow(2).mean() / target.pow(2).mean())


def best_linear(target: Tensor, columns: list[Tensor]) -> float:
    """R^2 of the best linear combination of these columns.

    One scalar weight per column, shared across the spatial axes -- the axes of
    an isotropic melt are equivalent, so a per-axis weight would be fitting noise.
    """
    design = torch.stack([column.reshape(-1) for column in columns], dim=1)
    weights = torch.linalg.lstsq(design, target.reshape(-1, 1)).solution
    return r2_uncentred(target, sum(weights[index, 0] * column for index, column in enumerate(columns)))


def linear_ladder(samples: Samples) -> dict[str, float]:
    """Every rung of the floor ladder for one set of samples."""
    target = samples.target
    return {
        # Predict zero acceleration, i.e. carry the current velocity forward.
        # Note this is *not* the frozen-positions baseline of `metrics.rollout`.
        "frozen": 0.0,
        "affine": best_linear(target, [samples.affine]),
        "kinematic": best_linear(target, [samples.affine, samples.velocity, samples.velocity_prev]),
        "force": best_linear(target, [samples.affine, samples.velocity, samples.velocity_prev, samples.force]),
        "velocity": best_linear(target, [samples.velocity, samples.velocity_prev]),
        "velocity_force": best_linear(target, [samples.velocity, samples.velocity_prev, samples.force]),
    }


def position_errors(samples: Samples, ladder: dict[str, float]) -> dict[str, float]:
    """Each rung's one-step position error, in the trajectory's length unit.

    The integrator adds the current velocity to both the prediction and the
    truth, so the position error of a one-step prediction is exactly the
    regression residual on the acceleration:

        x_hat(t+1) - x(t+1) = a_hat(t) - a(t)

    and since the ladder reports `R^2 = 1 - mse / mean(target^2)`, the mean
    squared position error is `(1 - R^2) * mean(target^2)`. Deriving it that way
    rather than refitting guarantees it agrees with the R^2 printed beside it.
    """
    scale = float(samples.target.pow(2).mean())
    errors = {"frozen_position_mse": scale, "frozen_position_rms": scale**0.5}
    for rung in ("affine", "kinematic", "force", "velocity"):
        if rung in ladder:
            mse = (1.0 - ladder[rung]) * scale
            errors[f"{rung}_position_mse"] = mse
            errors[f"{rung}_position_rms"] = max(mse, 0.0) ** 0.5
    return errors
