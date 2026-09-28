"""Whether a rolled-out trajectory carries the right mechanics.

A model can track positions well and still get the material's response wrong, so
the position metrics are not enough. This measures the stress the predicted
configurations actually exert, through the force field's own virial, and compares
its response to strain with the ground truth's.

Three scores per system, each a slope of stress against axial strain: `sxx_slope`
along the compressed axis, `syz_slope` averaged over the two transverse axes, and
their `ratio`. At fixed transverse box the ratio is `nu / (1 - nu)`, so it is the
one stress number that still says something about Poisson's ratio on a dataset
whose transverse box is clamped. Across systems each becomes an R^2 against the
truth. Alongside them `stress_rel_mse` is the median over systems of
`MSE(sigma_pred, sigma_true) / Var(sigma_true)` on the three diagonal components.

One thing here is not optional. A rollout's frames carry whatever neighbour list
the model was trained on, and a virial summed over a truncated list is not a
slightly different number -- it is the truncation. Every frame is rebuilt at the
force field's own cutoff before the sum, and `physics` refuses the job otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from ..graph.build import prepare_frames
from .. import physics
from .rollout import r2_centred

__all__ = ["slope", "stress_along", "stress_entry", "summarise_stress", "stress_dim"]


def slope(x: Tensor, y: Tensor) -> float:
    """Least-squares slope of `y` against `x`, through their means."""
    x, y = x.double(), y.double()
    centred = x - x.mean()
    return float((centred * (y - y.mean())).sum() / (centred * centred).sum())


def stress_along(frames: list[Data], potential, *, device: str = "cpu") -> tuple[Tensor, Tensor]:
    """`(axial strain, virial stress)` for a list of raw frames.

    The frames are prepared at `potential.physics`, the full force-field cutoff,
    whatever radius the model happened to be trained on.
    """
    full = potential.physics
    prepared = prepare_frames([frame.cpu() for frame in frames], full, pair_edges_on="all")
    reference = prepared[0].box_tensor[0]
    strains, stresses = [], []
    for graph in prepared:
        graph = graph.to(device)
        strains.append((graph.box_tensor[0] - reference) / reference)
        stresses.append(physics.virial_stress(graph, full).cpu())
    return torch.stack(strains), torch.stack(stresses)


def stress_dim(stress: Tensor) -> int:
    """Spatial dimension behind a `[frames, components]` stress tensor.

    `virial_stress` returns the diagonal first, then the off-diagonals, so the
    component count is `d + d(d-1)/2`: 3 in two dimensions, 6 in three.
    """
    count = stress.shape[1]
    dim = round((-1 + (1 + 8 * count) ** 0.5) / 2)
    if dim + dim * (dim - 1) // 2 != count:
        raise ValueError(f"{count} stress components match no spatial dimension.")
    return dim


def _transverse(stress: Tensor, driven_axis: int = 0) -> Tensor:
    """Mean normal stress over the axes that are not driven."""
    dim = stress_dim(stress)
    axes = [axis for axis in range(dim) if axis != driven_axis]
    return sum(stress[:, axis] for axis in axes) / len(axes)


def stress_entry(
    predicted: list[Data],
    truth: list[Data],
    potential,
    *,
    history: int,
    sample_stride: int = 5,
    device: str = "cpu",
) -> dict:
    """Stress response of one rolled-out system against its ground truth."""
    reached = len(predicted)
    indices = list(range(history, reached, sample_stride))
    if len(indices) < 3:
        return {}

    true_strain, true_stress = stress_along([truth[i] for i in indices], potential, device=device)
    pred_strain, pred_stress = stress_along([predicted[i] for i in indices], potential, device=device)

    dim = stress_dim(true_stress)
    entry = {
        "sxx_slope": (slope(true_strain, true_stress[:, 0]), slope(pred_strain, pred_stress[:, 0])),
        "syz_slope": (
            slope(true_strain, _transverse(true_stress)),
            slope(pred_strain, _transverse(pred_stress)),
        ),
        # The normal components only; the shears are not part of the response
        # being compared and would dilute the ratio.
        "stress_mse": float(((pred_stress[:, :dim] - true_stress[:, :dim]) ** 2).mean()),
        "stress_var": float(true_stress[:, :dim].var()),
    }
    entry["ratio"] = tuple(
        transverse / axial if abs(axial) > 1e-12 else float("nan")
        for transverse, axial in zip(entry["syz_slope"], entry["sxx_slope"])
    )
    return entry


def summarise_stress(entries: list[dict]) -> dict:
    """Aggregate per-system stress entries into the reported metrics."""
    entries = [entry for entry in entries if entry]
    if len(entries) < 3:
        return {}
    summary: dict = {}
    for key in ("sxx_slope", "syz_slope", "ratio"):
        truth = [entry[key][0] for entry in entries]
        predicted = [entry[key][1] for entry in entries]
        summary[f"{key}_r2"] = r2_centred(truth, predicted)
        summary[f"{key}_truth_mean"] = sum(truth) / len(truth)
        summary[f"{key}_pred_mean"] = sum(predicted) / len(predicted)
    ratios = torch.tensor(
        [entry["stress_mse"] / entry["stress_var"] for entry in entries if entry["stress_var"] > 0]
    )
    # Median, not mean: one system that blew up would otherwise set the number.
    summary["stress_rel_mse"] = float(ratios.median()) if ratios.numel() else float("nan")
    return summary
