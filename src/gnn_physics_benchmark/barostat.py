"""The box during a rollout: Lx is driven, the free axes follow a barostat.

The networks were compressed along the driven axis at a constant rate while
LAMMPS barostatted the free axes to zero stress, so the transverse box is a
response of the material. A rollout reproduces both: the driven edge keeps the
per-frame increment of the last two seed frames, and every free edge is a
damped piston pushed by the virial pressure of the *predicted* positions.
Poisson's ratio is then read off the box, exactly as from the ground truth.

Ported from `GNNInverseDesign/barostat_utils.py`. The pressure uses the LAMMPS
harmonic convention, `E = K (r - l0)^2` and so a bond force of `2K (r - l0)`,
because the registered coupling and damping were fitted with it. The kinetic
pressure is left out: at the datasets' temperature of 1e-7 it is negligible.

Parameters are per dataset and must reproduce its box. `resolve` checks them
before a run by driving the barostat with ground-truth positions and comparing
the transverse box with the true one; a dataset without parameters gets them
fitted by a grid search over the same check.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor
from torch_geometric.data import Data

from .graph.build import minimum_image

__all__ = ["Barostat", "BoxDriver", "box_error", "fit", "resolve", "TOLERANCE"]

#: Largest relative box error, with ground-truth positions, accepted for a run.
#: The registered parameters sit at 0.02-0.05 and recover Poisson's ratio with
#: R^2 >= 0.997 on held-out networks.
TOLERANCE = 0.1


class Barostat:
    """How the box of one dataset moves: driven axis, free axes and piston constants.

    `params` holds `coupling` and `damping`, the piston mass and friction per
    particle, and `dt`, the LAMMPS timestep. A dataset whose transverse box is
    clamped has no free axes and needs no parameters.
    """

    def __init__(self, entry, potential, params: dict | None):
        self.driven_axis = entry.driven_axis
        self.free_axes = entry.free_axes
        self.potential = potential
        self.params = params

    def drive(self, seed: list[Data]) -> "BoxDriver":
        """The box state of one rollout continuing from `seed`."""
        return BoxDriver(self, seed)


class BoxDriver:
    """The current box and the velocity of its free edges, advanced one frame at a time."""

    def __init__(self, barostat: Barostat, seed: list[Data]):
        self.barostat = barostat
        self.box = seed[-1].box_tensor.clone()
        self.delta = seed[-1].box_tensor[barostat.driven_axis] - seed[-2].box_tensor[barostat.driven_axis]
        if barostat.free_axes:
            self.step = (int(seed[-1].time) - int(seed[-2].time)) * barostat.params["dt"]
            boxes = [frame.box_tensor for frame in seed[-3:]]
            # Second-order backward difference of the last three seed boxes.
            self.velocity = (3 * boxes[2] - 4 * boxes[1] + boxes[0]) / (2 * self.step)

    def advance(self, position: Tensor, bond_index: Tensor, bond_attr: Tensor) -> Tensor:
        """The box of the next frame, given that frame's positions."""
        barostat = self.barostat
        box = self.box.clone()
        box[barostat.driven_axis] += self.delta
        if barostat.free_axes:
            nodes = position.shape[0]
            mass = barostat.params["coupling"] * nodes * self.step**2
            friction = barostat.params["damping"] * nodes * self.step
            # The current box sets the minimum image and the volume, as in the reference.
            pressure = virial_pressure(position.detach(), bond_index, bond_attr, self.box, barostat.potential)
            velocity = self.velocity.clone()
            for axis in barostat.free_axes:
                area = torch.prod(self.box) / self.box[axis]
                force = pressure[axis] * area - friction * self.velocity[axis]
                velocity[axis] = self.velocity[axis] + force / mass * self.step
                box[axis] = self.box[axis] + velocity[axis] * self.step
            self.velocity = velocity
        self.box = box
        return box


def virial_pressure(position: Tensor, bond_index: Tensor, bond_attr: Tensor, box: Tensor, potential) -> Tensor:
    """Diagonal virial pressure of the harmonic bonds, one value per axis.

    `bond_index` lists every bond once, as stored. Positive is outward.
    """
    source, target = bond_index
    vector = minimum_image(position[source] - position[target], box)
    distance = torch.linalg.vector_norm(vector, dim=1)
    force = -2.0 * potential.stiffness(bond_attr) * (distance - potential.rest_length(bond_attr))
    return (force.unsqueeze(1) * vector**2 / distance.unsqueeze(1)).sum(dim=0) / torch.prod(box)


def box_error(barostat: Barostat, trajectories: list[list[Data]], window: int) -> float:
    """Relative error of the free box edges when the barostat is driven by true positions.

    Each trajectory is seeded with its first `window` frames and advanced through
    all the frames it has. The error is the RMS deviation from the true edges divided by the RMS
    of their true change since the seed, so 0 is perfect and 1 is no better than
    a box that never moved.
    """
    error = change = 0.0
    for trajectory in trajectories:
        driver = barostat.drive(trajectory[:window])
        seed = trajectory[window - 1].box_tensor
        for frame in trajectory[window:]:
            box = driver.advance(frame.pos, frame.edge_index, frame.edge_attr)
            for axis in barostat.free_axes:
                error += float(box[axis] - frame.box_tensor[axis]) ** 2
                change += float(frame.box_tensor[axis] - seed[axis]) ** 2
    return math.sqrt(error / change) if change > 0 and math.isfinite(error) else float("inf")


def fit(entry, potential, trajectories: list[list[Data]], window: int, *, dt: float = 0.01) -> dict:
    """Coupling and damping that best reproduce the true box, by a coarse then a fine grid."""

    def score(coupling: float, damping: float) -> float:
        params = {"coupling": coupling, "damping": damping, "dt": dt}
        return box_error(Barostat(entry, potential, params), trajectories, window)

    def search(couplings, dampings):
        return min(itertools.product(couplings, dampings), key=lambda pair: score(*pair))

    decades = lambda low, high, count: [10 ** (low + (high - low) * i / (count - 1)) for i in range(count)]
    coupling, damping = search(decades(-4, 1, 11), decades(-5, 0, 11))
    lc, ld = math.log10(coupling), math.log10(damping)
    coupling, damping = search(decades(lc - 0.5, lc + 0.5, 9), decades(ld - 0.5, ld + 0.5, 9))
    return {"coupling": coupling, "damping": damping, "dt": dt}


def resolve(entry, potential, trajectories: list[list[Data]], window: int, *, systems: int = 10) -> tuple[Barostat, float]:
    """The dataset's barostat, checked against ground truth before it is trusted.

    Uses the first `systems` trajectories, every frame they have. Over a short
    stretch the error is dominated by the estimated initial box velocity, so check
    over at least the rollout horizon. Registered parameters that miss the
    true box by more than `TOLERANCE` stop the run. A dataset without registered
    parameters says so and has them fitted here; register the printed values so
    the next run does not search again.
    """
    if not entry.free_axes:
        return Barostat(entry, potential, None), 0.0
    if potential is None:
        raise ValueError(
            f"dataset {entry.key!r} has a free box axis but declares no force field, "
            "so the barostat has no pressure to follow."
        )
    sample = trajectories[:systems]
    params = entry.barostat
    if params is None:
        print(f"[barostat] no barostat parameters are registered for {entry.key!r}; "
              f"fitting them on {len(sample)} ground-truth trajectories.", flush=True)
        params = fit(entry, potential, sample, window)
        print(f"[barostat] fitted {params}; add them as `barostat=` to the {entry.key!r} "
              "registry entry.", flush=True)
    barostat = Barostat(entry, potential, params)
    error = box_error(barostat, sample, window)
    print(f"[barostat] {entry.key}: relative box error {error:.4f} over {len(sample[0]) - window} steps "
          f"with ground-truth positions (tolerance {TOLERANCE}).", flush=True)
    if error > TOLERANCE:
        raise ValueError(
            f"the barostat parameters {params} do not reproduce the {entry.key!r} box: relative "
            f"error {error:.4f} > {TOLERANCE}. Refit them with `gnn-bench barostat {entry.key} --fit`."
        )
    return barostat, error
