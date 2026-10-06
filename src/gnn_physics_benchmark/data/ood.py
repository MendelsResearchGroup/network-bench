"""An explicit split: train on the highest 30% of Poisson's ratios, test on the rest.

The ranking is the Poisson's ratio the dataset's registry records for each
system. Validation spans the full range of ratios, so checkpoint selection sees
networks like the ones the model is tested on as well as the ones it trained on.
"""

from dataclasses import replace
import math

import torch

from . import registry
from ..training.config import RunConfig


def prepare_ood(config: RunConfig, train_networks: int = 100, val_networks: int = 50) -> tuple[RunConfig, dict]:
    """Rank every system by its registry Poisson's ratio and split the ranking."""
    entry = registry.get(config.dataset)
    spec = config.split
    systems = entry.manifest(spec.manifest) if spec.manifest else entry.systems()
    ratios = entry.poisson_ratios()
    ranked = [{"system": stem, "poisson_ratio": ratios[stem]} for stem in sorted(systems)]
    return split_ranked(config, ranked, train_networks, val_networks)


def split_ranked(config: RunConfig, ranked: list[dict], train_networks: int = 100,
                 val_networks: int = 50) -> tuple[RunConfig, dict]:
    """Assign a ranking: training from the upper 30%, test from the lower 70%.

    The upper pool and the lower pool are each shuffled with the split seed.
    Training takes the first `train_networks` of the upper pool and test at most
    100 of the lower one. Validation covers the full range of ratios: 30% of its
    `val_networks` (rounded up) come from what the upper pool has left, as many
    as there are, and the rest from what the lower pool has left.
    """
    spec = config.split
    ranked = sorted(ranked, key=lambda row: (-row["poisson_ratio"], row["system"]))
    upper_count = math.ceil(0.3 * len(ranked))
    generator = torch.Generator().manual_seed(spec.seed)

    def shuffled(stems: list[str]) -> list[str]:
        return [stems[index] for index in torch.randperm(len(stems), generator=generator).tolist()]

    upper = shuffled([row["system"] for row in ranked[:upper_count]])
    lower = shuffled([row["system"] for row in ranked[upper_count:]])
    upper_val = upper[train_networks:][: math.ceil(0.3 * val_networks)]
    lower_val = lower[100:][: val_networks - len(upper_val)]
    explicit = {"train": upper[:train_networks], "val": upper_val + lower_val, "test": lower[:100]}
    split = replace(spec, explicit=explicit, manifest=None, sizes=None, limit=None, ood=True)
    audit = {"dataset": config.dataset, "ratio_source": "registry",
             "upper_fraction": 0.3, "train_networks": train_networks, "val_networks": val_networks,
             "test_limit": 100, "cutoff_poisson_ratio": ranked[upper_count - 1]["poisson_ratio"],
             "counts": {part: len(stems) for part, stems in explicit.items()},
             "unused": upper[train_networks + len(upper_val):] + lower[100 + len(lower_val):],
             "ranking": ranked}
    return replace(config, split=split), audit
