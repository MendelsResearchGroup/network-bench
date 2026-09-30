"""Choosing training windows, and preparing each one only once.

A window is `history + 1` consecutive frames the model reads plus the frames it
is scored against. Preparing one does not depend on the weights, so the cache
prepares it in the first epoch and serves it from then on.
"""

from __future__ import annotations

import random

from torch import Tensor
from torch_geometric.data import Data

from ..graph.features import prepare_window
from ..graph.potential import KGPotential
from ..interfaces.graph import InputGraphSpec

__all__ = ["window_starts", "target_positions", "WindowCache"]


def window_starts(num_frames: int, span: int, count: int, mode: str, rng: random.Random) -> list[int]:
    """Frame indices a window of `span` frames can start at.

    `span` counts every frame the window touches, inputs and targets together,
    so `start + span <= num_frames` always holds.
    """
    last = num_frames - span
    if last < 0:
        raise ValueError(f"a window of {span} frames does not fit in a {num_frames}-frame trajectory.")

    count = min(count, last + 1)
    if mode == "head":
        return list(range(count))
    if mode == "random":
        return sorted(rng.sample(range(last + 1), count))
    if mode == "spread":
        if count == 1:
            return [0]
        step = last / (count - 1)
        return sorted({int(round(index * step)) for index in range(count)})
    raise ValueError(f"unknown window mode {mode!r}.")


def target_positions(trajectory: list[Data], index: int, device: str) -> Tensor:
    """The positions to score against, without moving the stored frame.

    `Data.to()` migrates a store in place and returns the same object, so asking
    a stored frame for a device copy would move the dataset one frame at a time.
    Only the positions are needed, so only they are copied.
    """
    return trajectory[index].x.to(device)


def _to_device(graph: Data, device: str) -> Data:
    """A new `Data` holding this graph's tensors on `device`."""
    return Data(
        **{
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in graph.stores[0].items()
        }
    )


class WindowCache:
    """Prepared input windows, kept between epochs.

    With `enabled=False` every window is prepared on demand instead.
    """

    def __init__(self, spec: InputGraphSpec, potential: KGPotential | None, *, device: str = "cpu", enabled: bool = True):
        self.spec = spec
        self.potential = potential
        self.device = device
        self.enabled = enabled
        self._entries: dict[tuple, list[Data]] = {}

    def _prepare(self, trajectory: list[Data], start: int) -> list[Data]:
        frames = trajectory[start : start + self.spec.window_length]
        prepared = prepare_window(frames, self.spec, self.potential)
        return [_to_device(graph, self.device) for graph in prepared]

    def get(self, key: tuple, trajectory: list[Data], start: int) -> list[Data]:
        """The prepared window starting at `start`.

        A fresh list every time, because multi-step training shifts its window
        in place; the graphs inside are shared and read-only.
        """
        if not self.enabled:
            return self._prepare(trajectory, start)
        if key not in self._entries:
            self._entries[key] = self._prepare(trajectory, start)
        return list(self._entries[key])

    def __repr__(self) -> str:
        return f"WindowCache: {len(self._entries)} windows"
