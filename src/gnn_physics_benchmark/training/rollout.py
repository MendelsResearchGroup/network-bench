"""Rolling a model forward on its own predictions.

Because a model returns a frame in the raw dataset schema, a rollout is just the
training window slid forward: append the predicted frame, drop the oldest, and
prepare the window exactly as training prepares it. There is no separate
inference path to keep in step with the training path, which is where the
reference implementation had to duplicate its box handling.

The box is not predicted. Compression is imposed externally, so `box_mode`
decides how it moves: `"deform_x"` keeps the per-frame increment seen in the last
two seed frames, reproducing `fix deform ... erate` at a constant rate with the
transverse edges frozen; `"none"` holds it still.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..graph.features import build_input_graph, prepare_window
from ..graph.potential import KGPotential
from ..interfaces.graph import InputGraphSpec
from ..interfaces.model import frame_from_positions

__all__ = ["diverged", "advance_box", "prediction_frames", "rollout"]


def diverged(frame: Data) -> bool:
    """Has the rollout left the region where the graph still means anything?

    A free-running rollout of an under-trained model does blow up, and it blows
    up in ways that make the next step ill-defined rather than merely wrong: a
    non-finite position, or a box edge that has collapsed or gone negative. The
    caller stops and reports the truncation instead of propagating a NaN or
    tripping the minimum-image check.
    """
    return (
        not torch.isfinite(frame.pos).all()
        or not torch.isfinite(frame.box_tensor).all()
        or bool((frame.box_tensor <= 0).any())
    )


def advance_box(frame: Data, box_delta_x: torch.Tensor, box_mode: str) -> Data:
    """Move the box one rollout step."""
    if box_mode == "deform_x":
        box = frame.box_tensor.clone()
        box[0] = box[0] + box_delta_x
        frame.box_tensor = box
    return frame


def prediction_frames(graph: Data, predicted: Data, stride: int, box_delta_x: torch.Tensor, box_mode: str) -> list[Data]:
    """Fill skipped frames using the predicted endpoint, never future truth.

    This keeps the next input's velocity history on consecutive stored-frame
    intervals even when the model predicts two intervals in one call.
    """
    frames = []
    for offset in range(1, stride):
        position = graph.pos + (predicted.pos - graph.pos) * (offset / stride)
        frame = frame_from_positions(graph, position, offset)
        frames.append(advance_box(frame, offset * box_delta_x, box_mode))
    frames.append(advance_box(predicted, stride * box_delta_x, box_mode))
    return frames


def rollout(
    model,
    seed: list[Data],
    num_steps: int,
    spec: InputGraphSpec,
    potential: KGPotential | None,
    *,
    box_mode: str = "deform_x",
    device: str = "cpu",
) -> list[Data]:
    """Roll `model` forward `num_steps` frames from a seed of raw frames.

    `num_steps` counts input-frame intervals. A prediction stride of 2 needs
    half as many model calls; intermediate returned frames are interpolated.

    Returns CPU frames: the seed followed by everything the model produced, so the
    result indexes like the ground-truth trajectory it is compared against. A
    short return means the rollout diverged and was stopped.
    """
    trajectory = [frame.clone().to(device) for frame in seed]
    window = trajectory[-spec.window_length :]
    box_delta_x = window[-1].box_tensor[0] - window[-2].box_tensor[0]

    was_training = getattr(model, "training", False)
    model.eval()
    try:
        with torch.no_grad():
            stride = spec.prediction_stride
            for step in range(0, num_steps, stride):
                graph = build_input_graph(prepare_window(window, spec, potential), spec, potential)
                frames = prediction_frames(graph, model(graph), stride, box_delta_x, box_mode)
                predicted = frames[-1]
                if diverged(predicted):
                    print(f"[warning] rollout diverged after {step} steps.")
                    break
                frames = [frame.detach() for frame in frames[:num_steps - step]]
                window = (window + frames)[-spec.window_length:]
                trajectory.extend(frames)
    finally:
        if was_training:
            model.train()
    return [frame.cpu() for frame in trajectory]
