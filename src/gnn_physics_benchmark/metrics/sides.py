"""Side-node estimator from MendelsResearchGroup/graph_utils.

Functions copied unchanged from pratio.py at
734c6b81b5cddc539d1fad2e465e333f45ee5a90.
"""
from __future__ import annotations
from typing import Any
import numpy as np

def _box_to_dict(box: Any) -> dict[str, float]:
    if isinstance(box, dict):
        out = {k: float(v) for k, v in box.items()}
    else:
        out = {}
        for k in ("x1", "x2", "y1", "y2", "x", "y"):
            if hasattr(box, k):
                out[k] = float(getattr(box, k))
    if "x" not in out and "x1" in out and "x2" in out:
        out["x"] = out["x2"] - out["x1"]
    if "y" not in out and "y1" in out and "y2" in out:
        out["y"] = out["y2"] - out["y1"]
    return out


def _graph_box(graph: Any) -> dict[str, float]:
    box = graph.box if hasattr(graph, "box") else graph["box"]
    return _box_to_dict(box)


def _pos_np(graph: Any) -> np.ndarray:
    return graph.x[:, :2].detach().cpu().numpy()


def _pick_quantile_indices(distances: np.ndarray, quantile: float, eps: float) -> np.ndarray:
    thr = float(np.quantile(distances, quantile))
    idx = np.where(distances <= thr + eps)[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(distances))], dtype=int)
    return idx


def directional_side_indices_from_box(
    first_graph: Any, quantile: float = 0.10, eps: float = 1e-12
) -> dict[str, np.ndarray]:
    """Select node subsets near each side: left/right/top/bottom."""
    p = _pos_np(first_graph)
    b = _graph_box(first_graph)
    x = p[:, 0]
    y = p[:, 1]

    d_left = np.abs(x - b["x1"])
    d_right = np.abs(b["x2"] - x)
    d_bottom = np.abs(y - b["y1"])
    d_top = np.abs(b["y2"] - y)

    return {
        "left": _pick_quantile_indices(d_left, quantile=quantile, eps=eps),
        "right": _pick_quantile_indices(d_right, quantile=quantile, eps=eps),
        "bottom": _pick_quantile_indices(d_bottom, quantile=quantile, eps=eps),
        "top": _pick_quantile_indices(d_top, quantile=quantile, eps=eps),
    }


def _mean_axis(graph: Any, axis: int, node_idx: np.ndarray) -> float:
    p = _pos_np(graph)
    return float(np.mean(p[node_idx, axis]))


def _p_ratio_from_dx_dy(dx: float, dy: float, eps: float) -> float:
    if (not np.isfinite(dx)) or (not np.isfinite(dy)):
        return float("nan")
    if abs(dx) <= eps or abs(dy) <= eps:
        return float("nan")

    p1 = -(dy / dx)
    p2 = -(dx / dy)
    return float(p1 if abs(p1) < abs(p2) else p2)


def calc_p_ratio_rollout_sides(
    rollout: list,
    last_index: int = -1,
    side_idx: dict[str, np.ndarray] | None = None,
    side_quantile: float = 0.10,
    eps: float = 1e-12,
) -> float:
    """
    Directional rollout p-ratio estimate:
    dx from left/right strips and dy from top/bottom strips.
    """
    if side_idx is None:
        side_idx = directional_side_indices_from_box(rollout[0], quantile=side_quantile, eps=eps)

    first = rollout[0]
    last = rollout[last_index]

    w0 = _mean_axis(first, 0, side_idx["right"]) - _mean_axis(first, 0, side_idx["left"])
    w1 = _mean_axis(last, 0, side_idx["right"]) - _mean_axis(last, 0, side_idx["left"])
    h0 = _mean_axis(first, 1, side_idx["top"]) - _mean_axis(first, 1, side_idx["bottom"])
    h1 = _mean_axis(last, 1, side_idx["top"]) - _mean_axis(last, 1, side_idx["bottom"])

    dx = float(w1 - w0)
    dy = float(h1 - h0)
    return _p_ratio_from_dx_dy(dx, dy, eps=eps)
