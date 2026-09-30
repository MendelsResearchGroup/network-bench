"""Read and write benchmark results as plain JSON files."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from .. import models
from ..training.config import RunConfig

__all__ = [
    "results_root",
    "difficulty_path",
    "run_dir",
    "write_result",
    "read_result",
    "load_results",
    "export_results",
    "write_export",
]


def results_root(root: str | Path | None = None) -> Path:
    """Where results live: `root` if given, else `results/` beside the repository."""
    if root is not None:
        return Path(root).expanduser()
    # src/gnn_physics_benchmark/results/cache.py -> repository root
    return Path(__file__).resolve().parents[3] / "results"


def difficulty_path(dataset: str, root: str | Path | None = None) -> Path:
    return results_root(root) / dataset / "difficulty.json"


def run_hash(config: RunConfig, split: dict[str, list[str]]) -> str:
    payload = json.dumps({"config": config.to_dict(), "split": split}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def run_dir(config: RunConfig, split: dict[str, list[str]], root: str | Path | None = None) -> Path:
    base = results_root(root) / config.dataset / config.model
    digest = run_hash(config, split)
    return base / f"seed-{config.train.seed}_{digest}"


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
        (directory / "metrics.json").unlink(missing_ok=True)
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
    return [read_result(path.parent) for path in sorted(base.glob("*/seed-*/config.json"))]


def export_results(root: str | Path | None = None) -> dict:
    """Every finished run, across datasets, in the shape `docs/index.html` reads.

    Failed and unfinished runs are left out. Non-finite numbers become `None`,
    since JSON has no NaN.
    """
    runs = []
    for path in sorted(results_root(root).glob("*/*/seed-*/metrics.json")):
        result = read_result(path.parent)
        config, measured = result["config"], result["metrics"]
        train = config["train"]
        runs.append({
            "dataset": config["dataset"],
            "model": config["model"],
            "seed": train["seed"],
            "name": result["name"],
            "hyperparameters": {**models.defaults(config["model"]), **config["model_hyperparameters"]},
            "parameters": measured.get("parameters"),
            "split": {part: len(systems) for part, systems in result["split"].items()},
            "epochs": train["epochs"],
            "select_by": train.get("select_by"),
            "selected_epoch": measured.get("selected_epoch"),
            "val_score": measured.get(f"val_{train.get('select_by')}"),
            "rollout_steps": measured.get("requested_steps"),
            "relative_mse": measured.get("relative_mse"),
            "position_mse": measured.get("position_mse"),
            "diverged": measured.get("diverged"),
            "poisson_r2": {
                int(key.rsplit("_", 1)[1]): value
                for key, value in measured.items()
                if key.startswith("poisson_r2_")
            },
        })
    return {"runs": _finite(runs)}


def write_export(path: str | Path, root: str | Path | None = None) -> int:
    """Write `export_results` to `path`, atomically, so parallel jobs never leave half a file."""
    path = Path(path)
    payload = export_results(root)
    payload["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    os.replace(temporary, path)
    return len(payload["runs"])


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value
