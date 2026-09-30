"""Compare repeated runs with identical settings and data splits."""

import hashlib
import json
import math
from statistics import mean, stdev


def summarize_seeds(runs: list[dict]) -> list[dict]:
    """Group completed runs by everything except the training seed.

    Keep the resolved systems and evaluated split in the grouping key. Report
    finite counts per metric so missing or divergent scores stay visible.
    """
    groups = {}
    for run in runs:
        if "metrics" not in run or "failed" in run:
            continue
        config = {**run["config"], "train": dict(run["config"]["train"])}
        seed = config["train"].pop("seed")
        key = json.dumps([config, run["split"], run["metrics"]["split"]], sort_keys=True)
        group = groups.setdefault(key, {"model": config["model"], "name": hashlib.sha256(key.encode()).hexdigest()[:8], "runs": {}})
        group["runs"][seed] = run["metrics"]

    summaries = []
    for group in groups.values():
        scores = list(group["runs"].values())
        metrics = {}
        for key in sorted(set().union(*(score.keys() for score in scores))):
            values = [score[key] for score in scores if isinstance(score.get(key), (int, float))
                      and math.isfinite(score[key])]
            if values:
                metrics[key] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else None,
                                "n": len(values)}
        summaries.append({"model": group["model"], "name": group["name"],
                          "seeds": sorted(group["runs"]), "metrics": metrics})
    return summaries
