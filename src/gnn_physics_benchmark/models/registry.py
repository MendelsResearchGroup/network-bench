"""The model registry: a user-chosen key for every model.

Adding a model is two lines here plus the model itself. The key is what results
are cached under and what a comparison table is indexed by, so it should name the
*idea* -- "gns", "mlp" -- and leave the hyperparameters to `hyperparameters`,
which is hashed separately.

A registered factory is called as

    build(key, spec, target_scale, **hyperparameters) -> SimulatorModel

where `spec` is the run's `InputGraphSpec` and `target_scale` is the acceleration
normaliser the benchmark fitted for this dataset. A model that needs neither may
ignore them, but must accept them.

A model does not have to live in this repository. Register one from anywhere:

    from gnn_physics_benchmark.models import register
    register("my_model", MyModel)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..interfaces.graph import InputGraphSpec
from ..normalization import Normalizer
from .frozen import Frozen
from .gns import GNS
from .linear_floor import LinearFloor
from .mlp import NodeMLP
from .tiny_model import TinyVelocityMLP

__all__ = ["MODELS", "register", "build", "keys", "defaults"]

Factory = Callable[..., Any]

MODELS: dict[str, Factory] = {
    "gns": GNS,
    "mlp": NodeMLP,
    "frozen": Frozen,
    "linear_floor": LinearFloor,
    "tiny_mlp": TinyVelocityMLP,
}


def register(key: str, factory: Factory, *, overwrite: bool = False) -> None:
    """Make a model available under `key`."""
    if key in MODELS and not overwrite:
        raise KeyError(f"model key {key!r} is already registered; pass overwrite=True to replace it.")
    MODELS[key] = factory


def keys() -> list[str]:
    return sorted(MODELS)


def build(key: str, spec: InputGraphSpec, target_scale: Normalizer, **hyperparameters):
    """Instantiate a registered model."""
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; registered: {keys()}.")
    return MODELS[key](spec, target_scale, **hyperparameters)


def defaults(key: str) -> dict:
    """The hyperparameters `key` would be built with if none were given.

    Read off the factory's signature, so a model declares its defaults in one
    place -- its own constructor -- rather than repeating them in the registry.
    """
    import inspect

    signature = inspect.signature(MODELS[key])
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is not inspect.Parameter.empty
    }
