"""Training a registered model on a registered dataset.

Two loops share almost everything:

* **one-step** feeds a true window and scores the next frame. This is what the
  loss is designed for and what most runs use.
* **multi-step** feeds the model's own predictions back in and scores every
  rolled-out step, so the graphs it trains on are the graphs it will meet at
  inference. The rollout schedule grows the horizon as training proceeds.

Neither injects input noise; that is deliberately out of this version.
"""

from __future__ import annotations

import random
import math
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from ..graph.features import build_input_graph, potential_for, prepare_window
from ..metrics.rollout import rollout_errors, summarise_rollouts
from .config import RunConfig, TrainSpec
from .loss import prediction_loss
from .rollout import diverged, prediction_frames, rollout
from .windows import WindowCache, target_positions, window_starts

__all__ = ["train", "validate_one_step", "validate_rollout"]


def _optimizer(model, spec: TrainSpec):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if spec.weight_decay:
        return torch.optim.AdamW(parameters, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    return torch.optim.Adam(parameters, lr=spec.learning_rate)


def _step_optimizer(model, optimizer, spec: TrainSpec) -> None:
    if spec.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
    optimizer.step()
    optimizer.zero_grad()


def _freeze_input_normalizers(model, epoch: int, spec: TrainSpec) -> None:
    if epoch != spec.freeze_input_norm_epoch:
        return
    for normalizer in model.input_normalizers():
        normalizer.freeze()


def _rollout_steps_for(epoch: int, schedule: tuple[tuple[int, int], ...]) -> int:
    steps = schedule[0][1]
    for first_epoch, value in schedule:
        if epoch >= first_epoch:
            steps = value
    return steps


def validate_one_step(model, trajectories, graph_spec, potential, spec: TrainSpec, scale) -> float:
    """Mean one-step loss over the first windows of every validation system."""
    rng = random.Random(spec.seed)
    stride = graph_spec.prediction_stride
    span = graph_spec.window_length + stride
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for trajectory in trajectories:
            for start in window_starts(len(trajectory), span, spec.windows_per_sim, spec.window_mode, rng):
                window = trajectory[start : start + graph_spec.window_length]
                graph = build_input_graph(prepare_window(window, graph_spec, potential), graph_spec, potential)
                graph = graph.to(spec.device)
                target = target_positions(trajectory, start + graph_spec.window_length - 1 + stride, spec.device)
                total += float(
                    prediction_loss(model, model(graph), graph, target, scale, kind=spec.loss, accumulate=False)
                )
                count += 1
    model.train()
    return total / max(count, 1)


def validate_rollout(model, trajectories, graph_spec, potential, spec: TrainSpec, *, driven_axis: int = 0) -> dict:
    """Roll the model out on every validation system and summarise the errors."""
    steps = spec.val_rollout_steps
    shortest = min((len(t) for t in trajectories), default=graph_spec.window_length)
    requested = min(steps, max(1, shortest - graph_spec.window_length))
    errors = []
    for trajectory in trajectories:
        available = len(trajectory) - graph_spec.window_length
        if available < 1:
            continue
        predicted = rollout(
            model,
            trajectory[: graph_spec.window_length],
            min(steps, available),
            graph_spec,
            potential,
            box_mode=spec.box_mode,
            device=spec.device,
        )
        errors.append(rollout_errors(predicted, trajectory, graph_spec.history, driven_axis=driven_axis))
    return summarise_rollouts(errors, requested)


def _report(epoch: int, spec: TrainSpec, train_loss: float, metrics: dict, seconds: float) -> str:
    line = f"epoch {epoch + 1:3d}/{spec.epochs}  train {train_loss:.4e}"
    if "val_loss" in metrics:
        line += f"  val {metrics['val_loss']:.4e}"
    if "relative_mse" in metrics:
        line += (
            f"  rollout {metrics['position_mse']:.3e}"
            f"  rel {metrics['relative_mse']:.3f}"
            f"  steps {metrics['steps_completed']:.0f}"
        )
        if metrics["diverged"]:
            line += f"  {metrics['diverged']} diverged"
    return line + f"  [{seconds:.1f}s]"


def train(
    model,
    data: dict[str, list[list[Data]]],
    config: RunConfig,
    scale,
    *,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train `model` and return its per-epoch history.

    `data` maps split names to lists of raw trajectories; `scale` is the frozen
    acceleration normaliser the model was built with.
    """
    spec, graph_spec = config.train, config.graph
    from ..data import registry

    entry = registry.get(config.dataset)
    potential = potential_for(graph_spec, entry)

    torch.manual_seed(spec.seed)
    rng = random.Random(spec.seed)
    optimizer = _optimizer(model, spec)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, spec.gamma)
    cache = WindowCache(graph_spec, potential, device=spec.device, enabled=spec.cache_windows)

    multi_step = spec.mode == "multi_step"
    span = graph_spec.window_length + spec.max_rollout_steps * graph_spec.prediction_stride
    history: dict = {"train_loss": [], "epochs": [], "early_stopped": False}
    best_score, stale_checks = None, 0
    selection = spec.selection_metric
    lower = selection is not None and ("loss" in selection or "mse" in selection)

    model.train()
    optimizer.zero_grad()

    for epoch in range(spec.epochs):
        started = time.perf_counter()
        _freeze_input_normalizers(model, epoch, spec)
        steps = _rollout_steps_for(epoch, spec.rollout_schedule) if multi_step else 1

        total_loss, samples, truncated = 0.0, 0, 0
        for index, trajectory in enumerate(data["train"]):
            starts = window_starts(len(trajectory), span, spec.windows_per_sim, spec.window_mode, rng)
            for position, start in enumerate(starts):
                loss, taken, stopped = _window_loss(
                    model, trajectory, start, steps, graph_spec, spec, potential, scale, cache, index
                )
                truncated += int(stopped)
                (loss / spec.accumulation_steps).backward()
                if (position + 1) % spec.accumulation_steps == 0:
                    _step_optimizer(model, optimizer, spec)
                total_loss += float(loss)
                samples += 1
            _step_optimizer(model, optimizer, spec)  # flush the tail of this system

        train_loss = total_loss / max(samples, 1)
        history["train_loss"].append(train_loss)

        metrics: dict = {}
        due = (epoch + 1) % spec.validate_every == 0 or epoch == spec.epochs - 1
        if due and data.get("val"):
            metrics["val_loss"] = validate_one_step(model, data["val"], graph_spec, potential, spec, scale)
            metrics.update(
                validate_rollout(model, data["val"], graph_spec, potential, spec, driven_axis=entry.driven_axis)
            )
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, config, scale, Path(save_dir) / f"epoch_{epoch + 1:04d}.pt")
            if spec.early_stopping_patience is not None:
                score = metrics[selection]
                improved = best_score is None or (score < best_score if lower else score > best_score)
                if math.isfinite(score) and improved:
                    best_score, stale_checks = score, 0
                else:
                    stale_checks += 1
                history["early_stopped"] = stale_checks >= spec.early_stopping_patience and epoch + 1 < spec.epochs
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = train_loss
        if multi_step:
            metrics["rollout_steps"] = steps
            metrics["truncated"] = truncated
        history["epochs"].append(metrics)

        scheduler.step()
        if verbose:
            print(_report(epoch, spec, train_loss, metrics, time.perf_counter() - started), flush=True)
            if epoch == 0 and spec.cache_windows:
                print(f"  {cache}", flush=True)
        if history["early_stopped"]:
            if verbose:
                print(f"Early stopping: {selection} did not improve for {stale_checks} validation checks", flush=True)
            break

    history["cache"] = repr(cache)
    return history


def _window_loss(model, trajectory, start, steps, graph_spec, spec, potential, scale, cache, index):
    """Loss for one training window: one step, or a short rollout of `steps`."""
    window = cache.get(("train", index, start), trajectory, start)
    graph = build_input_graph(window, graph_spec, potential)
    stride = graph_spec.prediction_stride
    target = target_positions(trajectory, start + graph_spec.window_length - 1 + stride, spec.device)
    loss = prediction_loss(model, (frame := model(graph)), graph, target, scale, kind=spec.loss)

    if steps == 1:
        return loss, 1, False

    box_delta_x = graph.box_tensor[0] - window[-2].box_tensor[0]
    raw = [g.clone().to(spec.device) for g in trajectory[start : start + graph_spec.window_length]]
    taken = 1
    for step in range(1, steps):
        frames = prediction_frames(graph, frame, stride, box_delta_x, spec.box_mode)
        if diverged(frames[-1]):
            return loss / taken, taken, True
        frames = [frame.detach() if spec.detach_rollout else frame for frame in frames]
        raw = (raw + frames)[-graph_spec.window_length:]
        graph = build_input_graph(prepare_window(raw, graph_spec, potential), graph_spec, potential)
        target = target_positions(trajectory, start + graph_spec.window_length - 1 + (step + 1) * stride, spec.device)
        frame = model(graph)
        loss = loss + prediction_loss(model, frame, graph, target, scale, kind=spec.loss)
        taken += 1
    return loss / taken, taken, False


def save_checkpoint(model, config: RunConfig, scale, path: str | Path) -> None:
    """Save weights together with the config that produced them.

    The config is stored so that a checkpoint is self-describing. The reference
    implementation stored weights alone, which forced its scoring tool to
    reverse-engineer the architecture from tensor shapes.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "target_scale": scale.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path):
    """`(state_dict, target_scale_state, RunConfig)` from a saved checkpoint."""
    payload = torch.load(path, weights_only=False, map_location="cpu")
    return payload["model"], payload["target_scale"], RunConfig.from_dict(payload["config"])
