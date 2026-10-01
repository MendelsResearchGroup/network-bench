"""JSON settings for data splits, training, and a single benchmark run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..interfaces.graph import InputGraphSpec

__all__ = ["SplitSpec", "TrainSpec", "RunConfig"]


@dataclass(frozen=True)
class SplitSpec:
    """Explicit system names or a seeded split; membership is saved with results."""

    manifest: str | None = "clean"
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    sizes: tuple[int, int, int] | None = None  # Counts override ratios.
    seed: int = 42
    limit: int | None = None
    explicit: dict[str, list[str]] | None = None  # Names override generated splits.
    first_frame: int = 0
    frame_stride: int = 1
    max_frames: int | None = None
    ood: bool = False  # Highest-Poisson train/validation pool; lower-Poisson test.


@dataclass(frozen=True)
class TrainSpec:
    """Training settings; defaults also work for CPU runs."""

    epochs: int = 40
    learning_rate: float = 1e-3
    gamma: float = 0.995  # Learning-rate multiplier per epoch.
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    accumulation_steps: int = 10
    windows_per_sim: int = 15
    window_mode: str = "head"  # head, spread, random
    loss: str = "mse"  # mse, huber; applied to standardised acceleration
    freeze_input_norm_epoch: int = 5
    target_scale_windows: int | None = None  # None uses every training window.
    validate_every: int = 5
    val_rollout_steps: int = 50
    select_by: str | None = None  # None uses val_loss with early stopping, otherwise the last epoch.
    early_stopping_patience: int | None = None  # Validation checks without improvement; None disables stopping.
    stress_metrics: bool = False
    mode: str = "one_step"  # one_step, multi_step
    rollout_schedule: tuple[tuple[int, int], ...] = ((0, 1),)  # (first epoch, steps)
    detach_rollout: bool = False
    box_mode: str = "deform_x"  # deform_x, none
    device: str = "cpu"
    seed: int = 0
    cache_windows: bool = True

    @property
    def max_rollout_steps(self) -> int:
        return max(steps for _, steps in self.rollout_schedule) if self.mode == "multi_step" else 1

    @property
    def selection_metric(self) -> str | None:
        return self.select_by or ("val_loss" if self.early_stopping_patience is not None else None)


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
        payload = asdict(self)
        # Preserve the cache keys of existing normal runs.
        if not self.split.ood:
            payload["split"].pop("ood")
        if self.train.early_stopping_patience is None:
            payload["train"].pop("early_stopping_patience")
        return payload

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        payload = dict(payload)
        graph = InputGraphSpec(**payload.pop("graph", {}))
        split = SplitSpec(**payload.pop("split", {}))
        train = TrainSpec(**payload.pop("train", {}))
        return cls(graph=graph, split=split, train=train, **payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
