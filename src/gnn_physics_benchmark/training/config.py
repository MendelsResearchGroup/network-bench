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

    batch_size: int = 1
    accumulation_steps: int = 10
    """Optimizer steps are taken every `accumulation_steps` batches. Batches are
    formed inside one trajectory and never across two, so the bead count is
    constant within a batch and its mean loss is exactly the mean of its
    windows' losses."""

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
    cache_limit_mb: float = 4096.0

    def __post_init__(self) -> None:
        if self.mode not in ("one_step", "multi_step"):
            raise ValueError(f"mode must be 'one_step' or 'multi_step', got {self.mode!r}.")
        if self.window_mode not in ("head", "spread", "random"):
            raise ValueError(f"window_mode must be head, spread or random, got {self.window_mode!r}.")
        if self.box_mode not in ("none", "deform_x"):
            raise ValueError(f"box_mode must be 'none' or 'deform_x', got {self.box_mode!r}.")

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
        split = SplitSpec(**_tuples(payload.pop("split", {}) or {}, ("ratios",)))
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
