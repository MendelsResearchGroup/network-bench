"""Dataset difficulty and model performance metrics.

Three different R^2 conventions live here and must stay distinct, or numbers
quietly stop meaning what they say:

* `ladder.r2_uncentred` -- `1 - mse / mean(target^2)`. For the acceleration
  floors, where the target's mean is physically zero.
* `rollout.r2_centred` -- the usual coefficient of determination against the
  variance. For one scalar per system: a Poisson's ratio, a stress slope.
* `metrics.rollout.summarise_rollouts["relative_mse"]` is a ratio of means over
  systems, not an R^2 at all.
"""

from .difficulty import (
    alpha_from_positions,
    describe,
    floor_over_time,
    persistence_from_positions,
    quantisation_noise,
    samples_from_positions,
    trajectory_arrays,
)
from .ladder import Samples, best_linear, linear_ladder, position_errors, r2_uncentred
from .rollout import mean_poisson_ratio, r2_centred, rollout_errors, summarise_rollouts
from .stress import slope, stress_along, stress_dim, stress_entry, summarise_stress

__all__ = [
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
    "rollout_errors",
    "summarise_rollouts",
    "mean_poisson_ratio",
    "r2_centred",
    "stress_along",
    "stress_entry",
    "summarise_stress",
    "slope",
    "stress_dim",
]
