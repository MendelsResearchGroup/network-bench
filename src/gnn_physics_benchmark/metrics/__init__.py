"""Model performance metrics. Dataset difficulty lives in `gnn_physics_benchmark.difficulty`.

Two conventions, kept distinct:

* `rollout.r2_centred` -- the usual coefficient of determination against the
  variance. For one scalar per system: a Poisson's ratio, a stress slope.
* `metrics.rollout.summarise_rollouts["relative_mse"]` is a ratio of means over
  systems, not an R^2 at all.
"""

from .rollout import box_poisson_ratio, r2_centred, rollout_errors, summarise_rollouts
from .stress import slope, stress_along, stress_dim, stress_entry, summarise_stress

__all__ = [
    "rollout_errors",
    "summarise_rollouts",
    "box_poisson_ratio",
    "r2_centred",
    "stress_along",
    "stress_entry",
    "summarise_stress",
    "slope",
    "stress_dim",
]
