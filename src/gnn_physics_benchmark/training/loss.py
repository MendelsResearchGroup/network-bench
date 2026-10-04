"""The benchmark's loss: what every model is scored on.

The target is the one-step acceleration,

    a_t = (x_{t+1} - x_t) - (x_t - x_{t-1})

For prediction stride `s`, the effective acceleration target becomes
`(x_{t+s} - x_t - s * (x_t - x_{t-1})) / s**2`. The input history remains
consecutive; only the predicted endpoint moves farther ahead.

standardised by a scale fitted once, before training, on **exactly the windows
training will visit**. Fitting it up front rather than accumulating it during
training is what stops the model chasing a target that moves underneath it, and
it means every epoch's loss is quoted in the same units from the first step.

Fitting it on the right windows matters more than it sounds. In this data the
acceleration scale varies by two orders of magnitude *along a single
trajectory* -- on one system the target RMS is 2.5e-6 over the first twenty
frames and 6.6e-4 by frame seventy, as the accumulated strain starts to bite.
A scale drawn uniformly over the whole trajectory is then tens of times larger
than anything a `window_mode="head"` run actually sees, which both flatters the
loss and forces the model's decoder to work far from the O(1) range the
standardisation exists to put it in.

The same fixed scale divides both the prediction and the truth, so the number is
comparable across models whatever any of them does internally. It also gives the
loss a reading: with a z-scored target, the mean squared error is literally
`1 - R^2`, so 1.0 is "no better than predicting the mean" and 0.0 is perfect.

A model whose whole idea is a different target normalisation may define
`normalize_target(acceleration, accumulate=...)` and will be scored through that
instead -- deliberately off the shared scale.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor
from torch_geometric.data import Data

from ..interfaces.model import current_velocity, predicted_acceleration
from ..normalization import Normalizer
from .windows import window_starts

__all__ = ["acceleration_target", "fit_target_scale", "LOSSES", "prediction_loss"]


def acceleration_target(graph: Data, target_position: Tensor) -> Tensor:
    """The acceleration that takes the input graph's window to `target_position`."""
    stride = getattr(graph, "prediction_stride", 1)
    return (target_position - graph.pos - stride * current_velocity(graph)) / stride**2


def fit_target_scale(
    trajectories: list[list[Data]],
    graph_spec,
    train_spec,
    *,
    max_windows: int | None = None,
    device: str = "cpu",
) -> Normalizer:
    """Fit and freeze the acceleration scale on the windows training will read.

    The window starts are drawn the same way the training loop draws them -- same
    mode, same count, same seed, same order -- so the scale describes the targets
    the model is actually scored on. Under `window_mode="random"` that is exact
    for the first epoch and representative afterwards; under `"head"` and
    `"spread"` the starts never change, so it is exact throughout.

    Only positions are read, so no graph is prepared and this is cheap.
    `max_windows` caps the sample for a very large split; by default every window
    contributes.
    """
    window_length = graph_spec.window_length
    stride = graph_spec.prediction_stride
    span = window_length + train_spec.max_rollout_steps * stride
    scale = Normalizer(trajectories[0][0].x.shape[1]).to(device)
    rng = random.Random(train_spec.seed)

    windows = [
        (index, start)
        for index, trajectory in enumerate(trajectories)
        for start in window_starts(
            len(trajectory), span, train_spec.windows_per_sim, train_spec.window_mode, rng
        )
    ]
    if not windows:
        raise ValueError("no training window fits; the trajectories are shorter than the history.")
    if max_windows is not None and len(windows) > max_windows:
        windows = random.Random(train_spec.seed).sample(windows, max_windows)

    for index, start in windows:
        trajectory = trajectories[index]
        # Every target the window is scored against: one for a one-step run, and
        # one per rolled-out step for a multi-step one.
        for step in range(train_spec.max_rollout_steps):
            current_index = start + window_length - 1 + step * stride
            previous = trajectory[current_index - 1].x
            current = trajectory[current_index].x
            later = trajectory[current_index + stride].x
            scale(((later - current - stride * (current - previous)) / stride**2).to(device))
    scale.freeze()
    return scale


def _mse(predicted: Tensor, truth: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(predicted, truth)


def _huber(predicted: Tensor, truth: Tensor) -> Tensor:
    return torch.nn.functional.huber_loss(predicted, truth, delta=1.0)


LOSSES = {"mse": _mse, "huber": _huber}


def prediction_loss(
    model,
    frame: Data,
    graph: Data,
    target_position: Tensor,
    scale: Normalizer,
    *,
    kind: str = "mse",
    accumulate: bool = True,
) -> Tensor:
    """Score one predicted frame against the true next positions."""
    predicted = predicted_acceleration(frame, graph)
    truth = acceleration_target(graph, target_position)

    hook = getattr(model, "normalize_target", None)
    if hook is not None:
        return LOSSES[kind](hook(predicted, accumulate=False), hook(truth, accumulate=accumulate))
    # `scale` is frozen, so this is an exact inverse of the `scale.inverse()` a
    # model applies to its decoder output. Both losses depend only on the
    # difference of their arguments, so the mean cancels and only the division
    # by the standard deviation actually sets the scale.
    return LOSSES[kind](scale(predicted, accumulate=False), scale(truth, accumulate=False))
