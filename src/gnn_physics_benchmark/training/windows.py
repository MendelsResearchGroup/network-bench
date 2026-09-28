"""Choosing training windows, and not paying for them twice.

A window is `history + 1` consecutive frames the model reads plus the frames it
is scored against. Preparing one -- symmetrising the bonds and building a pair
neighbour list -- costs about as much as a small model's forward and backward
pass, and none of it depends on the weights. With the `"head"` or `"spread"`
window modes the frame indices are identical every epoch, so the whole cost is
paid in the first epoch and the cache serves every epoch after it.
"""

from __future__ import annotations

import random

import torch
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

    `limit_mb` bounds the store. Once it is reached the cache stops taking new
    entries and the rest are prepared on demand, so an unbounded `"random"`
    window mode degrades in speed rather than in memory.
    """

    def __init__(
        self,
        spec: InputGraphSpec,
        potential: KGPotential | None,
        *,
        device: str = "cpu",
        limit_mb: float = 4096.0,
        enabled: bool = True,
    ):
        self.spec = spec
        self.potential = potential
        self.device = device
        self.limit_bytes = limit_mb * 1024**2
        self.enabled = enabled
        self._entries: dict[tuple, list[Data]] = {}
        self._storages: set[int] = set()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.full = False

    def _prepare(self, trajectory: list[Data], start: int) -> list[Data]:
        frames = trajectory[start : start + self.spec.window_length]
        prepared = prepare_window(frames, self.spec, self.potential)
        return [_to_device(graph, self.device) for graph in prepared]

    def _cost(self, window: list[Data]) -> tuple[int, set[int]]:
        """Bytes this window would add, and the storages that would be new.

        Counted per underlying storage, not per tensor. The frames of one window
        share `atom_ids`, `atom_types`, `molecule_ids` and the whole bond
        topology, and overlapping windows share them too, so summing tensor
        sizes naively charges the same allocation many times over and the cache
        would stop well short of its limit.

        The pointers are returned rather than recorded, and the caller commits
        them only if it keeps the window: recording them here would leak, since
        a rejected window is freed straight away and its addresses can be handed
        out again to tensors that would then be counted as free.
        """
        total, seen = 0, set()
        for graph in window:
            for value in graph.stores[0].values():
                if not isinstance(value, Tensor):
                    continue
                pointer = value.untyped_storage().data_ptr()
                if pointer in self._storages or pointer in seen:
                    continue
                seen.add(pointer)
                total += value.numel() * value.element_size()
        return total, seen

    def get(self, key: tuple, trajectory: list[Data], start: int) -> list[Data]:
        """The prepared window starting at `start`, from the cache when possible.

        A fresh list is returned every time, because multi-step training shifts
        its window in place. The graphs inside are shared and read-only: every
        consumer either reads them or builds a new `Data`.
        """
        if not self.enabled:
            return self._prepare(trajectory, start)

        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return list(cached)

        self.misses += 1
        window = self._prepare(trajectory, start)
        size, storages = self._cost(window)
        if self.bytes + size <= self.limit_bytes:
            self._entries[key] = window
            self._storages |= storages
            self.bytes += size
        else:
            self.full = True
        return list(window)

    def clear(self) -> None:
        self._entries.clear()
        self._storages.clear()
        self.bytes = 0
        self.full = False

    def __repr__(self) -> str:
        total = self.hits + self.misses
        rate = 100 * self.hits / total if total else 0.0
        state = " (limit reached)" if self.full else ""
        return (
            f"WindowCache: {len(self._entries)} windows, {self.bytes / 1024**2:.0f} MiB, "
            f"{rate:.0f}% hit rate{state}"
        )
