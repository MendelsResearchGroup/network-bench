"""How hard a dataset is, before any model is trained on it.

Everything here reads positions and boxes as bare arrays -- `(frames, nodes, dim)`
and `(frames, dim)` -- so it needs no graph, no force field and no model. Run it
once per dataset and the numbers describe the data itself.

What to read first
------------------
`floor_model` is the linear floor matched to what a model is actually fed: past
residual velocities, no box. Then `r2_ceiling` beside it, because a floor only
means something relative to how much of the target is signal at all -- a floor of
0.6 says very different things under a ceiling of 0.65 and under one of 1.0.
`floor_share_of_ceiling` is their ratio: how much of the reachable signal a
linear model already takes.

The ceiling is real and this data sits close to it. Positions are stored in
float32 at O(3) sigma while the one-step target is O(3e-6), so the rounding of
the stored positions puts genuine noise into the target. `quantisation_noise`
estimates its RMS and `r2_ceiling` turns that into the best R^2 any predictor
could reach.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from .ladder import Samples, linear_ladder, position_errors

__all__ = [
    "trajectory_arrays",
    "samples_from_positions",
    "persistence_from_positions",
    "alpha_from_positions",
    "quantisation_noise",
    "describe",
    "floor_over_time",
]


def trajectory_arrays(trajectory: list[Data]) -> tuple[Tensor, Tensor]:
    """`(positions, boxes)` stacked over frames, as stored."""
    return (
        torch.stack([frame.x for frame in trajectory]),
        torch.stack([frame.box_tensor for frame in trajectory]),
    )


def _affine(positions: Tensor, boxes: Tensor) -> Tensor:
    """The displacement the box alone imposes, frame by frame."""
    return positions[:-1] * (boxes[1:] / boxes[:-1] - 1.0).unsqueeze(1)


def samples_from_positions(
    positions: Tensor,
    boxes: Tensor,
    *,
    count: int = 15,
    history: int = 6,
    forces: Tensor | None = None,
    offset: int | None = None,
    affine_axes: tuple[int, ...] | None = None,
) -> Samples:
    """One-step examples in the layout the ladder expects.

    The window bookkeeping mirrors training: head windows of length
    `history + 1` starting at 0, 1, ... put their targets on frames
    `history + 1 .. history + count`, so that is the span scored here. `offset`
    moves the span along the trajectory, which is how `floor_over_time` walks the
    same measurement down a trajectory; it must be at least 2, because every
    example needs two frames of history behind it.

    `affine_axes` chooses which axes' box motion is taken out of the two
    *velocity* columns: `None` means every axis, matching an input graph built
    with `velocity="residual"`; `()` means none, matching `velocity="total"`;
    `(0,)` removes only the driven axis. The target and the affine column are the
    same either way.

    Everything is promoted to float64 first. The target is a second difference of
    float32 positions five orders larger than itself, so the arithmetic below
    would otherwise measure its own rounding.
    """
    positions = positions.double()
    boxes = boxes.double()
    target, affine_delta, velocity, velocity_prev, force = [], [], [], [], []

    start = history if offset is None else offset
    if start < 2:
        raise ValueError(f"offset must be at least 2, got {start}.")
    if affine_axes is None:
        keep: Tensor | float = 1.0
    else:
        keep = torch.zeros(positions.shape[2], dtype=torch.float64)
        keep[list(affine_axes)] = 1.0

    for t in range(start, start + count):
        if t + 1 >= positions.shape[0]:
            break
        before, previous, current, later = positions[t - 2 : t + 2]
        box_before, box_previous, box_current, box_later = boxes[t - 2 : t + 2]

        affine_now = current * (box_later / box_current - 1.0)
        affine_before = previous * (box_current / box_previous - 1.0)
        # The affine part is scaled by the *earlier* frame's positions, because
        # that is where the bead sat when the box moved. Using `previous` here
        # instead silently changes the kinematic rung in the seventh decimal.
        affine_prev = before * (box_previous / box_before - 1.0)

        target.append((later - current) - (current - previous))
        affine_delta.append(affine_now - affine_before)
        velocity.append((current - previous) - keep * affine_before)
        velocity_prev.append((previous - before) - keep * affine_prev)
        force.append(torch.zeros_like(current) if forces is None else forces[t].double())

    if not target:
        raise ValueError(f"offset {start} plus count {count} needs more than {positions.shape[0]} frames.")
    return Samples(*(torch.cat(field) for field in (target, affine_delta, velocity, velocity_prev, force)))


def persistence_from_positions(positions: Tensor, boxes: Tensor, *, first: int = 0, count: int = 18) -> dict:
    """Mean cosine between consecutive residual velocities, and the affine share.

    The affine part is removed first: under a constant-rate deformation it is
    very nearly constant, so leaving it in makes every bead look perfectly
    persistent regardless of what the bead itself is doing.
    """
    positions = positions[first : first + count + 2].double()
    boxes = boxes[first : first + count + 2].double()
    if positions.shape[0] < 3:
        raise ValueError(f"need at least 3 frames, got {positions.shape[0]}.")

    affine = _affine(positions, boxes)
    raw = positions[1:] - positions[:-1]
    residual = raw - affine
    unit = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    cosine = (unit[:-1] * unit[1:]).sum(-1)
    return {
        "mean_persistence": float(cosine.mean()),
        "median_persistence": float(cosine.median()),
        "affine_share": float(affine.norm(dim=-1).mean() / raw.norm(dim=-1).mean()),
        "residual_rms": float(residual.pow(2).mean().sqrt()),
    }


def alpha_from_positions(
    positions: Tensor, boxes: Tensor, *, first: int = 0, count: int = 18, residual: bool = False
) -> dict:
    """The dynamics-complexity parameter of Shteingolts et al., JCP 163, 124115.

        alpha = (1 / (N L)) sum_i sum_t | v_{t+1,i} - v_{t,i} |

    Larger alpha means more complex dynamics. It is the mean *vector norm* of the
    frame-to-frame velocity change: the same acceleration the one-step target is
    built from, summarised differently.

    Two things make it awkward to compare across datasets, so both corrections
    are reported alongside rather than folded in. It is a norm, not a
    per-component figure, so for `d` roughly independent components it runs about
    `sqrt(d)` larger than one component -- `alpha_per_component` divides that
    out. And it is dimensional, scaling with how much the beads move at all --
    `alpha_over_speed` divides by the mean speed in the same window, giving a
    dimensionless "how much does the velocity change relative to its size" that
    survives a change of units.

    The paper takes velocities as stored, which is `residual=False`, and that is
    the number to quote. `residual=True` removes the affine box motion first, the
    convention every other metric here uses, and the two do not agree -- on the
    cold rubber by about 36%. Both are available so the gap is visible.
    """
    positions = positions[first : first + count + 2].double()
    boxes = boxes[first : first + count + 2].double()
    if positions.shape[0] < 3:
        raise ValueError(f"need at least 3 frames, got {positions.shape[0]}.")

    velocity = positions[1:] - positions[:-1]
    if residual:
        velocity = velocity - _affine(positions, boxes)

    change = velocity[1:] - velocity[:-1]
    norm = change.norm(dim=-1)
    return {
        "alpha": float(norm.mean()),
        "alpha_per_component": float(norm.mean() / positions.shape[2] ** 0.5),
        "alpha_over_speed": float(norm.mean() / velocity.norm(dim=-1).mean().clamp_min(1e-30)),
        "mean_speed": float(velocity.norm(dim=-1).mean()),
    }


def quantisation_noise(positions: Tensor) -> float:
    """RMS of the rounding noise the stored precision puts into the target.

    A stored position carries an error uniform on plus or minus half an ULP, so
    its variance is `ulp^2 / 12`. The target is the second difference
    `(1, -2, 1)`, whose coefficients sum in quadrature to `1 + 4 + 1 = 6`, so the
    noise variance in the target is `6 * ulp^2 / 12 = ulp^2 / 2` -- hence the
    factor of a half below. Pass the positions in their *stored* dtype; this is a
    property of how the data was written, not of how it is being processed.
    """
    ulp = torch.finfo(positions.dtype).eps * positions.double().abs().clamp_min(1e-30)
    return float((ulp.pow(2).mean() * 0.5).sqrt())


def describe(
    positions: Tensor, boxes: Tensor, *, history: int = 6, count: int = 15, forces: Tensor | None = None
) -> dict:
    """Every difficulty metric for one trajectory, as one flat row."""
    common = dict(count=count, history=history, forces=forces)
    samples = samples_from_positions(positions, boxes, **common)
    ladder = linear_ladder(samples)
    driven = linear_ladder(samples_from_positions(positions, boxes, affine_axes=(0,), **common))
    total = linear_ladder(samples_from_positions(positions, boxes, affine_axes=(), **common))

    row = {
        "nodes": int(positions.shape[1]),
        "frames": int(positions.shape[0]),
        "dim": int(positions.shape[2]),
        "target_rms": float(samples.target.pow(2).mean().sqrt()),
        # Three floors, because which one a model is held to depends on what it
        # is fed. `kinematic` has the box at the predicted frame and describes
        # the data; the `velocity` rungs are what an encoder actually sees, under
        # `velocity="residual"`, `"residual"` on the driven axis only, and `"total"`.
        "floor_affine": ladder["affine"],
        "floor_kinematic": ladder["kinematic"],
        "floor_velocity": ladder["velocity"],
        "floor_velocity_x": driven["velocity"],
        "floor_velocity_total": total["velocity"],
        # Share of the target's second moment the box accounts for: the part the
        # affine column hands `kinematic` and no encoder here can see.
        "affine_target_share": float(samples.affine.pow(2).mean() / samples.target.pow(2).mean()),
    }
    row.update(position_errors(samples, ladder))
    if forces is not None:
        row["floor_force"] = ladder["force"]

    # Bead displacement over exactly the frames training reads. If it is a tiny
    # fraction of a particle spacing then every window of a trajectory is the
    # same structure, and a model has nothing new to see from one to the next.
    span = positions[: history + count + 1]
    row["drift_over_windows"] = float((span[-1] - span[0]).pow(2).sum(-1).mean().sqrt())
    row.update(persistence_from_positions(positions, boxes, count=history + count))
    row.update(alpha_from_positions(positions, boxes, count=history + count))

    noise = quantisation_noise(positions)
    row["quant_noise"] = noise
    row["noise_over_signal"] = noise / row["target_rms"]
    row["r2_ceiling"] = max(0.0, 1.0 - (noise / row["target_rms"]) ** 2)
    row["floor_model"] = row["floor_velocity"] if row["dim"] == 3 else row["floor_velocity_x"]
    row["floor_share_of_ceiling"] = (
        row["floor_model"] / row["r2_ceiling"] if row["r2_ceiling"] > 0 else float("nan")
    )
    return row


def floor_over_time(
    positions: Tensor, boxes: Tensor, *, window: int = 15, stride: int | None = None, history: int = 6
) -> list[dict]:
    """The same measurement, repeated in successive windows along one trajectory.

    Training reads head windows, so every floor above describes the first twenty
    or so frames of a trajectory. That is a bet that the head is representative
    of the whole deformation. This walks the identical measurement down the
    trajectory so the bet can be checked, and so difficulty can be read against
    accumulated strain rather than against nothing.

    One row per window. `strain_x` is the cumulative compressive strain at the
    window's first target frame, which is the natural horizontal axis:
    trajectories of different lengths at the same strain rate then line up.
    """
    stride = window if stride is None else stride
    rows = []
    for offset in range(history, positions.shape[0] - 1 - window, stride):
        common = dict(count=window, history=history, offset=offset)
        samples = samples_from_positions(positions, boxes, **common)
        ladder = linear_ladder(samples)
        driven = linear_ladder(samples_from_positions(positions, boxes, affine_axes=(0,), **common))
        total = linear_ladder(samples_from_positions(positions, boxes, affine_axes=(), **common))
        # Over the frames this window spans, so the diagnostics describe the same
        # stretch of trajectory rather than drifting apart.
        moving = persistence_from_positions(positions, boxes, first=offset - 2, count=window + 1)
        alpha = alpha_from_positions(positions, boxes, first=offset - 2, count=window + 1)
        errors = position_errors(samples, ladder)
        rows.append(
            {
                "offset": offset,
                "first_target": offset + 1,
                "strain_x": float(1.0 - boxes[offset, 0] / boxes[0, 0]),
                "target_rms": float(samples.target.pow(2).mean().sqrt()),
                "floor_affine": ladder["affine"],
                "floor_kinematic": ladder["kinematic"],
                "floor_velocity": ladder["velocity"],
                "floor_velocity_x": driven["velocity"],
                "floor_velocity_total": total["velocity"],
                "kinematic_position_mse": errors["kinematic_position_mse"],
                "kinematic_position_rms": errors["kinematic_position_rms"],
                "velocity_position_rms": errors["velocity_position_rms"],
                "frozen_position_rms": errors["frozen_position_rms"],
                "mean_persistence": moving["mean_persistence"],
                "affine_share": moving["affine_share"],
                "alpha": alpha["alpha"],
                "alpha_over_speed": alpha["alpha_over_speed"],
            }
        )
    if not rows:
        raise ValueError(f"{positions.shape[0]} frames is too few for window {window} at history {history}.")
    return rows
