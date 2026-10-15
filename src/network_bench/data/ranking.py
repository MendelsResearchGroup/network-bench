"""Rank networks by their measured 100-step Poisson ratio."""
import math
from . import loading, registry
from ..metrics import position_poisson_ratio
from ..training.config import RunConfig


def rank_networks(config: RunConfig) -> list[dict]:
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
    return sorted(ranked, key=lambda row: (-row["poisson_ratio"], row["system"]))
