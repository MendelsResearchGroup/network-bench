"""Pieces shared by more than one baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from ..interfaces.graph import InputGraphSpec

__all__ = ["build_mlp", "AxisSharedEncoder", "node_extras"]


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    *,
    num_mlp: int = 3,
    layer_norm: bool = False,
) -> torch.nn.Sequential:
    """An MLP of `max(num_mlp, 2)` linear layers with ReLU between them.

    Weights are drawn from a Gaussian of width `1 / sqrt(fan_in)` rather than
    left at torch's uniform default, which is what the reference implementation
    trained with.
    """
    sizes = [in_size] + [hidden_size] * (max(num_mlp, 2) - 1) + [out_size]
    layers: list[torch.nn.Module] = []
    for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = torch.nn.Linear(fan_in, fan_out)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / fan_in**0.5)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0 / fan_in**0.5)
        layers.append(layer)
        if index < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    if layer_norm:
        layers.append(torch.nn.LayerNorm(normalized_shape=out_size))
    return torch.nn.Sequential(*layers)


def node_extras(graph, spec: InputGraphSpec) -> Tensor | None:
    """The per-axis channels beyond the velocity history, as `[N, dim, k]`.

    Both extras are vectors like a velocity, so they slot into the axis-shared
    layout rather than needing encoders of their own.
    """
    extras = []
    if spec.node_force:
        extras.append(graph.node_force)
    if spec.fractional_coordinates:
        extras.append(graph.fractional_coordinates)
    return torch.stack(extras, dim=2) if extras else None


class AxisSharedEncoder(torch.nn.Module):
    """Encode node features with one MLP shared across the spatial axes.

    The axes of an isotropic melt are equivalent, so sharing weights across them
    is both a parameter saving and a symmetry the model gets for free. The
    velocity history is laid out step-major -- `[v_t, v_{t-1}, ...]`, each a full
    3-vector -- precisely so that a single reshape puts the axes in the batch
    dimension and the history steps in the feature dimension.
    """

    def __init__(self, history: int, hidden_size: int, *, dim: int, num_mlp: int, extra_channels: int = 0):
        super().__init__()
        self.dim = dim
        self.history = history
        self.axis_mlp = build_mlp(history + extra_channels, hidden_size, hidden_size, num_mlp=num_mlp)

    def forward(self, x: Tensor, extra: Tensor | None = None) -> Tensor:
        # [N, history * dim] -> [N, history, dim] -> [N, dim, history]
        per_axis = x.view(x.size(0), self.history, self.dim).transpose(1, 2)
        if extra is not None:
            per_axis = torch.cat([per_axis, extra], dim=2)
        return self.axis_mlp(per_axis).reshape(x.size(0), -1)
