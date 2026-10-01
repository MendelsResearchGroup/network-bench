"""An explicit split from the highest 30% of ground-truth Poisson's ratios."""

from dataclasses import replace
import math

import torch

from . import loading, registry
from ..metrics import position_poisson_ratio
from ..training.config import RunConfig


def prepare_ood(config: RunConfig, train_networks: int = 30) -> tuple[RunConfig, dict]:
    """Rank at 100 predicted steps; train on 30 networks in the upper pool.

    Test on at most 100 of the lower 70%. Ranking uses the same position-based
    ratio, seed frame and frame stride as evaluation, never model predictions.
    """
    entry = registry.get(config.dataset)
    spec = config.split
    systems = entry.manifest(spec.manifest) if spec.manifest else entry.systems()
    history = config.graph.history
    ranked = []
    for index, stem in enumerate(sorted(systems)):
        frames = loading.load_trajectory(entry, stem, first_frame=spec.first_frame,
                                         frame_stride=spec.frame_stride, max_frames=history + 101)
        ratio = position_poisson_ratio(frames[history], frames[history + 100], driven_axis=entry.driven_axis)
        if not math.isfinite(ratio):
            raise ValueError(f"{config.dataset}/{stem}: cannot rank a non-finite Poisson's ratio")
        ranked.append({"system": stem, "poisson_ratio": ratio})
        if (index + 1) % 25 == 0:
            print(f"{config.dataset}: ranked {index + 1}/{len(systems)} systems", flush=True)
    return split_ranked(config, ranked, train_networks)


def split_ranked(config: RunConfig, ranked: list[dict], train_networks: int = 30) -> tuple[RunConfig, dict]:
    """Assign an already measured ranking without loading the trajectories again."""
    spec = config.split
    ranked = sorted(ranked, key=lambda row: (-row["poisson_ratio"], row["system"]))
    upper_count = math.ceil(0.3 * len(ranked))
    upper = [row["system"] for row in ranked[:upper_count]]
    generator = torch.Generator().manual_seed(spec.seed)
    order = torch.randperm(upper_count, generator=generator).tolist()
    upper = [upper[index] for index in order]
    lower = [row["system"] for row in ranked[upper_count:]]
    order = torch.randperm(len(lower), generator=generator).tolist()
    lower = [lower[index] for index in order]
    explicit = {"train": upper[:train_networks], "val": upper[train_networks:], "test": lower[:100]}
    split = replace(spec, explicit=explicit, manifest=None, sizes=None, limit=None, ood=True)
    audit = {"dataset": config.dataset, "ranking_steps": 100, "history_frames": config.graph.history + 1,
             "upper_fraction": 0.3, "train_networks": train_networks, "test_limit": 100,
             "cutoff_poisson_ratio": ranked[upper_count - 1]["poisson_ratio"],
             "counts": {part: len(stems) for part, stems in explicit.items()},
             "unused": lower[100:], "ranking": ranked}
    return replace(config, split=split), audit
