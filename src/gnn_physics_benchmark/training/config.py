"""The description of one benchmark run, and how it round-trips through JSON.

A `RunConfig` is the complete, self-contained answer to "what was measured": the
dataset key, the model key and its hyperparameters, how the input graph was
built, which systems were used, and how training was done. It is what the result
cache is keyed on and what gets written next to the metrics, so a number in a
results table can always be traced back to the thing that produced it.

Everything here is a plain dataclass of JSON-representable fields. There is no
default split: a run says which systems it used, because a comparison across
models is only meaningful if they saw the same ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..interfaces.graph import InputGraphSpec

__all__ = ["SplitSpec", "TrainSpec", "RunConfig"]


@dataclass(frozen=True)
class SplitSpec:
    """Which systems a run uses, and how their frames are read.

    Either name the systems outright with `explicit` -- the reproducible option,
    and what a published comparison should use -- or give a `manifest`, `ratios`
    and `seed` and let them be drawn. Either way the resolved membership lists go
    into the result key, so a regenerated manifest misses the cache instead of
    silently comparing unlike splits.
    """

    manifest: str | None = "clean"
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    sizes: tuple[int, int, int] | None = None
    """Train, val and test *counts* instead of `ratios`, e.g. (50, 50, 70): the
    first that many of the shuffled systems go to each part; the rest are unused."""
    seed: int = 42
    limit: int | None = None
    """Keep only the first `limit` systems before splitting. For smoke runs."""
    explicit: dict[str, list[str]] | None = None

    first_frame: int = 0
    frame_stride: int = 1
    """Keep every k-th frame. This changes what one step means, and therefore
    what the model is asked to predict, so it is part of the run's identity."""
    max_frames: int | None = None
    """Truncate each trajectory after this many kept frames."""


@dataclass(frozen=True)
class TrainSpec:
    """How the model is trained and validated."""

    mode: str = "one_step"
    """`"one_step"` scores the next frame from a true window. `"multi_step"`
    feeds the model's own predictions back in and scores each rolled-out step,
    so the graphs it trains on are the graphs it will see at inference."""

    epochs: int = 40
    learning_rate: float = 1e-3
    gamma: float = 0.995
    """Per-epoch multiplier of the exponential learning-rate decay."""
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    accumulation_steps: int = 10
    """Windows are trained one at a time; the optimizer steps every
    `accumulation_steps` windows, and at the end of each trajectory."""

    windows_per_sim: int = 15
    window_mode: str = "head"
    """`"head"` takes the first starting points, matching where a rollout begins;
    `"spread"` samples them evenly along the trajectory; `"random"` draws them."""

    loss: str = "mse"
    """`"mse"` or `"huber"`, on the standardised acceleration. Because the target
    is z-scored by a fixed scale, the MSE is literally `1 - R^2`."""

    freeze_input_norm_epoch: int = 5
    """Epoch at which a model's *input* normalisers stop accumulating. The target
    scale is fitted once before training and never moves, so it needs no such
    knob: a model is never chasing a target that shifts under it."""
    target_scale_windows: int | None = None
    """Cap on how many training windows the target scale is fitted on. `None`,
    the default, uses every window the run will visit, which is what makes the
    scale describe the targets the model is actually scored on."""

    validate_every: int = 5
    val_rollout_steps: int = 50
    select_by: str | None = None
    """A validation metric, e.g. `"poisson_r2_100"`: test with the checkpoint of the
    validation epoch that scored best on it (highest, or lowest for a `loss` or
    `mse`). `None` tests the last epoch. Only validation epochs have checkpoints,
    so `validate_every` sets how finely this can choose."""
    stress_metrics: bool = False
    """Also score the stress response on the test rollout (`sxx_slope_r2`,
    `ratio_r2`, `stress_rel_mse`). Needs the dataset's force field. Useful where the
    transverse box is clamped, so Poisson's ratio can only be read off the stress."""
    rollout_schedule: tuple[tuple[int, int], ...] = ((0, 1),)
    """`(first_epoch, steps)` ladder for multi-step training."""
    detach_rollout: bool = False
    """Cut the gradient between multi-step rollout steps. Off by default, so the
    gradient flows through the whole horizon; turn it on when the memory of a
    long horizon matters more than the through-time signal."""

    box_mode: str = "deform_x"
    """How the box moves during a rollout. `"deform_x"` keeps the per-frame
    increment seen in the last two seed frames, reproducing `fix deform ...
    erate`; `"none"` freezes it."""

    device: str = "cpu"
    seed: int = 0
    cache_windows: bool = True

    @property
    def max_rollout_steps(self) -> int:
        return max(steps for _, steps in self.rollout_schedule) if self.mode == "multi_step" else 1


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one benchmark run."""

    dataset: str
    model: str
    model_hyperparameters: dict = field(default_factory=dict)
    graph: InputGraphSpec = field(default_factory=InputGraphSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    train: TrainSpec = field(default_factory=TrainSpec)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=_jsonable))

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        payload = dict(payload)
        graph = InputGraphSpec(**payload.pop("graph", {}) or {})
        split = SplitSpec(**_tuples(payload.pop("split", {}) or {}, ("ratios", "sizes")))
        train = TrainSpec(**_schedule(_tuples(payload.pop("train", {}) or {}, ())))
        return cls(graph=graph, split=split, train=train, **payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def replace(self, **changes) -> "RunConfig":
        return replace(self, **changes)


def _jsonable(value):
    raise TypeError(f"{type(value).__name__} is not JSON-representable; keep RunConfig fields plain.")


def _tuples(payload: dict, keys: tuple[str, ...]) -> dict:
    """JSON turns tuples into lists; turn the ones that must be tuples back."""
    for key in keys:
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    return payload


def _schedule(payload: dict) -> dict:
    if "ratios" in payload and payload["ratios"] is not None:
        payload["ratios"] = tuple(payload["ratios"])
    if "rollout_schedule" in payload and payload["rollout_schedule"] is not None:
        payload["rollout_schedule"] = tuple(tuple(pair) for pair in payload["rollout_schedule"])
    return payload
