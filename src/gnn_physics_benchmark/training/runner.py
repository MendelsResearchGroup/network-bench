"""Turning a `RunConfig` into loaded data, a fitted scale and a built model.

Kept apart from the training loop so that the same setup can be reused by the
metrics, by an evaluation of a saved checkpoint, and by the command line.
"""

from __future__ import annotations

import torch

from .. import models
from ..data import loading, registry
from ..graph.features import potential_for
from .config import RunConfig
from .loss import fit_target_scale

__all__ = ["resolve_split", "load_split", "build", "Prepared"]


class Prepared:
    """Everything a run needs, assembled from its config."""

    def __init__(self, config: RunConfig):
        self.entry = registry.get(config.dataset)
        # The spatial dimension is the dataset's, not the run's, so a config need
        # not state it; resolving here means the saved config records what was used.
        self.config = config = config.replace(graph=config.graph.resolved(self.entry.schema))
        self.config = config
        self.potential = potential_for(config.graph, self.entry)
        self.split = resolve_split(config)
        self.data = load_split(config, self.split)
        self.scale = fit_target_scale(
            self.data["train"],
            config.graph,
            config.train,
            max_windows=config.train.target_scale_windows,
            device=config.train.device,
        )
        torch.manual_seed(config.train.seed)
        self.model = models.build(
            config.model, config.graph, self.scale, **config.model_hyperparameters
        ).to(config.train.device)

    def summary(self) -> str:
        sizes = {name: len(stems) for name, stems in self.split.items()}
        parameters = sum(p.numel() for p in self.model.parameters())
        return (
            f"dataset {self.config.dataset} split {sizes}\n"
            f"model {self.config.model} {self.model.hyperparameters} ({parameters} parameters)\n"
            f"graph {self.config.graph}\n"
            f"target sigma {[f'{v:.3e}' for v in self.scale.std().flatten().tolist()]}"
        )


def resolve_split(config: RunConfig) -> dict[str, list[str]]:
    spec = config.split
    return loading.resolve_split(
        registry.get(config.dataset),
        explicit=spec.explicit,
        manifest=spec.manifest,
        ratios=spec.ratios,
        seed=spec.seed,
        limit=spec.limit,
    )


def load_split(config: RunConfig, split: dict[str, list[str]]) -> dict[str, list[list]]:
    entry = registry.get(config.dataset)
    spec = config.split
    return {
        name: loading.load_trajectories(
            entry,
            stems,
            first_frame=spec.first_frame,
            frame_stride=spec.frame_stride,
            max_frames=spec.max_frames,
        )
        for name, stems in split.items()
    }


def build(config: RunConfig) -> Prepared:
    return Prepared(config)
