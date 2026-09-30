"""The results cache: one directory per benchmark cell.

    results/<dataset>/<model>/<readable_name>_<hash8>/
        config.json  split.json  metrics.json  history.json  checkpoints/  [failed.txt]
    results/<dataset>/difficulty.json

`readable_name` names only what differs from the defaults, so an ordinary run
has a short name and an unusual one announces what is unusual. `hash8` is taken
over the whole config *including the resolved list of systems in each split*, so
regenerating a manifest misses the cache instead of silently comparing unlike
runs. A failure writes `failed.txt` and no `metrics.json`, so a rerun retries it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import models
from ..training.config import RunConfig

__all__ = [
    "results_root",
    "difficulty_path",
    "run_dir",
    "run_exists",
    "write_result",
    "read_result",
    "load_results",
]


def results_root(root: str | Path | None = None) -> Path:
    """Where results live: `root` if given, else `results/` beside the repository."""
    if root is not None:
        return Path(root).expanduser()
    # src/gnn_physics_benchmark/results/cache.py -> repository root
    return Path(__file__).resolve().parents[3] / "results"


def difficulty_path(dataset: str, root: str | Path | None = None) -> Path:
    return results_root(root) / dataset / "difficulty.json"


def _flatten(payload: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _short(value) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_short(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


#: Already in the directory path (`dataset`, `model`) or filled in from the dataset
#: when a run is resolved (`dim`, `edge_width`), so naming them says nothing. They
#: are still in the hash.
NOT_NAMED = ("dataset", "model", "graph.dim", "graph.edge_width")


def readable_name(config: RunConfig) -> str:
    """`key=value` for every field that differs from the defaults, or `default`."""
    actual = _flatten(config.to_dict())
    default = _flatten(RunConfig(dataset=config.dataset, model=config.model).to_dict())
    default.update({f"model_hyperparameters.{k}": v for k, v in models.defaults(config.model).items()})
    changed = [
        f"{name.rsplit('.', 1)[-1]}={_short(value)}"
        for name, value in sorted(actual.items())
        if name not in NOT_NAMED and default.get(name, object()) != value
    ]
    name = "_".join(changed) or "default"
    return "".join(c if c.isalnum() or c in "=-._" else "-" for c in name)[:120]


def run_hash(config: RunConfig, split: dict[str, list[str]]) -> str:
    payload = json.dumps({"config": config.to_dict(), "split": split}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def run_dir(config: RunConfig, split: dict[str, list[str]], root: str | Path | None = None) -> Path:
    return (
        results_root(root)
        / config.dataset
        / config.model
        / f"{readable_name(config)}_{run_hash(config, split)}"
    )


def run_exists(config: RunConfig, split: dict[str, list[str]], root: str | Path | None = None) -> bool:
    return (run_dir(config, split, root) / "metrics.json").is_file()


def write_result(
    directory: Path,
    config: RunConfig,
    split: dict[str, list[str]],
    *,
    metrics: dict | None = None,
    history: dict | list | None = None,
    traceback_text: str | None = None,
) -> None:
    """Write one cell. With `traceback_text` it is a failure: `failed.txt`, no metrics."""
    directory.mkdir(parents=True, exist_ok=True)
    config.to_json(directory / "config.json")
    (directory / "split.json").write_text(json.dumps(split, indent=1))
    if traceback_text is not None:
        (directory / "failed.txt").write_text(traceback_text)
        return
    (directory / "failed.txt").unlink(missing_ok=True)
    if history is not None:
        (directory / "history.json").write_text(json.dumps(history, indent=1, default=float))
    (directory / "metrics.json").write_text(json.dumps(metrics or {}, indent=1, default=float))


def read_result(directory: Path) -> dict:
    """`{"name", "directory", "config", "split", "metrics"?, "history"?, "failed"?}`."""
    result = {"name": directory.name, "directory": str(directory)}
    for key in ("config", "split", "metrics", "history"):
        path = directory / f"{key}.json"
        if path.is_file():
            result[key] = json.loads(path.read_text())
    failed = directory / "failed.txt"
    if failed.is_file():
        result["failed"] = failed.read_text()
    return result


def load_results(dataset: str, root: str | Path | None = None) -> list[dict]:
    """Every cached cell of one dataset, failed ones included, sorted by model then name."""
    base = results_root(root) / dataset
    if not base.is_dir():
        return []
    return [read_result(path.parent) for path in sorted(base.glob("*/*/config.json"))]
