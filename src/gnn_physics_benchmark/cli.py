"""The `gnn-bench` command line.

    gnn-bench datasets                     what is registered, and what it is
    gnn-bench schema                       the raw frame schema and the input graph
    gnn-bench models                       registered models and their defaults
    gnn-bench difficulty <dataset>         measure a dataset before training on it
    gnn-bench train <config.json>          train, evaluate and cache one run
    gnn-bench evaluate <run_dir>           re-evaluate a cached run from its checkpoint
    gnn-bench report <dataset>             compare every cached run, side by side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import difficulty
from . import evaluate as evaluation
from . import models
from .data import registry
from .interfaces.graph import INPUT_GRAPH_SCHEMA, InputGraphSpec
from .interfaces.raw import describe_schema
from .results import cache
from .training import RunConfig

__all__ = ["main"]

#: Columns of the comparison table, in order: what to read first, first.
REPORT_COLUMNS = (
    ("relative_mse", "rel.MSE", "{:.4f}"),
    ("position_mse", "pos.MSE", "{:.3e}"),
    ("frozen_mse", "frozen", "{:.3e}"),
    ("steps_completed", "steps", "{:.0f}"),
    ("diverged", "div", "{:.0f}"),
    ("poisson_r2_10", "nu R2@10", "{:.3f}"),
    ("poisson_r2_50", "nu R2@50", "{:.3f}"),
    ("poisson_r2_100", "nu R2@100", "{:.3f}"),
    ("ratio_r2", "ratio R2", "{:.3f}"),
    ("sxx_slope_r2", "sxx R2", "{:.3f}"),
    ("stress_rel_mse", "str.relMSE", "{:.3e}"),
    ("parameters", "params", "{:.0f}"),
)


def _format(value, spec: str) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and value != value:
            return "nan"
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)] if rows else [
        len(h) for h in headers
    ]
    line = "  ".join(header.rjust(width) for header, width in zip(headers, widths))
    out = [line, "-" * len(line)]
    out += ["  ".join(str(cell).rjust(width) for cell, width in zip(row, widths)) for row in rows]
    return "\n".join(out)


def cmd_datasets(args) -> None:
    for key in registry.keys():
        entry = registry.get(key)
        present = entry.root().is_dir()
        print(f"{key}")
        print(f"  directory   {entry.root()} {'' if present else '(NOT FOUND)'}")
        print(f"  frames      {entry.frames} per system, {entry.interval} MD steps apart")
        print(f"  dim         {entry.schema.dim}   edge columns {entry.schema.edge_columns}")
        print(f"  driven axis {entry.driven_axis}   free axes {entry.free_axes or 'none (transverse box clamped)'}")
        print(f"  potential   {entry.potential}")
        if present:
            print(f"  systems     {len(entry.systems())} on disk")
            for name in entry.manifests:
                print(f"  manifest    {name}: {len(entry.manifest(name))} systems")
        print(f"\n  {entry.description}\n")
        if entry.field_notes:
            print(f"  Notes: {entry.field_notes}\n")


def cmd_schema(args) -> None:
    keys = [args.dataset] if args.dataset else registry.keys()
    for key in keys:
        entry = registry.get(key)
        print(f"=== {key} ===")
        print("RAW FRAME -- what a stored trajectory holds, and what a model must return\n")
        print(describe_schema(entry.schema))
        print()
    print("\nINPUT GRAPH -- what a model is handed\n")
    for name, shape, meaning in INPUT_GRAPH_SCHEMA:
        print(f"{name:22s} {shape:18s} {meaning}")
    print(f"\ndefault InputGraphSpec: {InputGraphSpec()}")
    print("(`dim` is filled in from the dataset when a run is resolved.)")


def cmd_models(args) -> None:
    for key in models.keys():
        print(f"{key:14s} {models.defaults(key)}")


def cmd_difficulty(args) -> None:
    result = difficulty.measure_difficulty(
        args.dataset,
        systems=args.systems,
        manifest=args.manifest,
        history=args.history,
        count=args.count,
        max_frames=args.max_frames,
    )
    path = cache.difficulty_path(args.dataset, args.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))

    median = result["median"]
    print(f"{args.dataset}: {result['systems']} systems, manifest {result['manifest']!r}\n")
    print("Read floor_model first, then r2_ceiling beside it: a floor only means")
    print("something relative to how much of the target is signal at all.\n")
    headline = [
        "floor_model", "r2_ceiling", "floor_share_of_ceiling", "floor_kinematic", "floor_affine",
        "affine_target_share", "target_rms", "frozen_position_rms", "velocity_position_rms",
        "drift_over_windows", "mean_persistence", "alpha_over_speed", "noise_over_signal",
    ]
    for key in headline:
        print(f"  {key:24s} {median[key]:,.6g}")
    print(f"\nfull per-system rows written to {path}")


def cmd_train(args) -> None:
    config = RunConfig.from_json(args.config)
    # One config can serve every dataset and model: these swap them in. A model
    # swapped in runs with its own default hyperparameters.
    if args.dataset:
        config = config.replace(dataset=args.dataset)
    if args.model:
        config = config.replace(model=args.model, model_hyperparameters={})
    # `--set model_hyperparameters.hidden_dim=64` changes one field of the config.
    payload = config.to_dict()
    for item in args.set:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = payload
        for part in path:
            node = node[part]
        try:
            node[last] = json.loads(value)
        except json.JSONDecodeError:
            node[last] = value
    config = RunConfig.from_dict(payload)
    torch.set_num_threads(args.threads)
    result = evaluation.run(config, root=args.root, resume=not args.force, split=args.split)
    print()
    print(json.dumps(result.get("metrics", {}), indent=1))


def cmd_evaluate(args) -> None:
    from .training import build, load_checkpoint

    directory = Path(args.run_dir)
    config = RunConfig.from_json(directory / "config.json")
    torch.set_num_threads(args.threads)
    checkpoints = sorted((directory / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint under {directory / 'checkpoints'}.")

    prepared = build(config)
    state, _, _ = load_checkpoint(checkpoints[-1])
    prepared.model.load_state_dict(state)
    print(f"{checkpoints[-1].name} on split {args.split!r}\n")
    print(json.dumps(evaluation.evaluate_model(prepared, split=args.split), indent=1))


def cmd_report(args) -> None:
    runs = cache.load_results(args.dataset, args.root)
    if not runs:
        raise SystemExit(f"no cached runs for {args.dataset} under {cache.results_root(args.root)}.")

    headers = ["model", "run"] + [label for _, label, _ in REPORT_COLUMNS]
    rows = []
    for run in runs:
        measured = run.get("metrics", {})
        config = run.get("config", {})
        rows.append(
            [config.get("model", "?"), run["name"]]
            + [_format(measured.get(key), spec) for key, _, spec in REPORT_COLUMNS]
        )
    print(f"{args.dataset}: {len(runs)} cached runs\n")
    print(_table(rows, headers))
    print("\nrel.MSE is pos.MSE / frozen, so 1.0 is no better than every bead standing still.")
    print("nu R2 is NaN wherever the transverse box is clamped; ratio R2 is the stress-based")
    print("Poisson's ratio, which is measurable there.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gnn-bench", description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="results directory (default: results/ beside the repo)")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="registered datasets").set_defaults(func=cmd_datasets)
    schema = sub.add_parser("schema", help="the raw frame and input graph schemas")
    schema.add_argument("dataset", nargs="?", default=None)
    schema.set_defaults(func=cmd_schema)
    sub.add_parser("models", help="registered models").set_defaults(func=cmd_models)

    difficulty = sub.add_parser("difficulty", help="measure a dataset's difficulty")
    difficulty.add_argument("dataset")
    difficulty.add_argument("--systems", type=int, default=20)
    difficulty.add_argument("--manifest", default="clean")
    difficulty.add_argument("--history", type=int, default=3)
    difficulty.add_argument("--count", type=int, default=15)
    difficulty.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    difficulty.set_defaults(func=cmd_difficulty)

    train = sub.add_parser("train", help="train, evaluate and cache one run")
    train.add_argument("config")
    train.add_argument("--split", default="test")
    train.add_argument("--dataset", help="override the config's dataset")
    train.add_argument("--model", help="override the config's model (with its default hyperparameters)")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="change one config field, e.g. train.epochs=80; repeatable")
    train.add_argument("--force", action="store_true", help="rerun even if already cached")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="re-evaluate a cached run from its last checkpoint")
    evaluate.add_argument("run_dir")
    evaluate.add_argument("--split", default="test")
    evaluate.set_defaults(func=cmd_evaluate)

    report = sub.add_parser("report", help="compare cached runs for one dataset")
    report.add_argument("dataset")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
