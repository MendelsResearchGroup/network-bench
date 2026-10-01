"""Running mean/std normalisation, shared by the models and by the loss.

The quantities in this problem span five orders of magnitude -- positions are
O(3) sigma, one-step accelerations O(3e-5) -- so nothing trains without being put
on a common scale first. A `Normalizer` accumulates first and second moments over
the batches it is shown, standardises with them, and is frozen part-way through
training so that the scale a loss is quoted in stops moving.

Unlike the reference implementation this keeps its statistics in registered
buffers rather than plain attributes, so `state_dict` carries them, `.to(device)`
moves them and a checkpoint needs no special handling.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["Normalizer"]


class Normalizer(torch.nn.Module):
    """Standardise a tensor by statistics accumulated over training batches.

    `freeze()` stops the accumulation. Freezing matters more than it looks: while the
    statistics move, so does the scale the loss is divided by, and two epochs'
    losses are not comparable.
    """

    def __init__(self, size: int, *, std_epsilon: float = 1e-8):
        super().__init__()
        self.size = size
        self.register_buffer("frozen", torch.tensor(False))
        self.register_buffer("std_epsilon", torch.tensor(float(std_epsilon)))
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("accumulations", torch.tensor(0.0))
        self.register_buffer("total", torch.zeros(1, size))
        self.register_buffer("total_squared", torch.zeros(1, size))

    def forward(self, data: Tensor, *, accumulate: bool = True) -> Tensor:
        if accumulate and not bool(self.frozen):
            self._accumulate(data.detach())
        return (data - self.mean()) / self.std()

    def inverse(self, normalized: Tensor) -> Tensor:
        return normalized * self.std() + self.mean()

    def freeze(self) -> None:
        self.frozen.fill_(True)

    def _accumulate(self, data: Tensor) -> None:
        self.total += data.sum(dim=0, keepdim=True)
        self.total_squared += (data**2).sum(dim=0, keepdim=True)
        self.count += data.shape[0]
        self.accumulations += 1

    def _safe_count(self) -> Tensor:
        return torch.clamp_min(self.count, 1.0)

    def mean(self) -> Tensor:
        return self.total / self._safe_count()

    def std(self) -> Tensor:
        variance = self.total_squared / self._safe_count() - self.mean() ** 2
        return torch.maximum(variance.clamp_min(0.0).sqrt(), self.std_epsilon)

    def extra_repr(self) -> str:
        return f"size={self.size}, frozen={bool(self.frozen)}, batches={int(self.accumulations)}"
