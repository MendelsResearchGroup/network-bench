"""The public, dependency-light interfaces of the benchmark.

This subpackage is what an external user imports:

    pip install git+https://github.com/<org>/gnn_physics_benchmark
    from gnn_physics_benchmark.interfaces import SimulatorModel, integrate_acceleration

It deliberately imports nothing beyond `torch` and `torch_geometric`, so a model
author can read and satisfy the contract without pulling in the training harness,
the metrics or the plotting extras.

Three things to read, in order:

* `raw` -- what a stored trajectory frame holds, and what a model must return.
* `graph` -- what a model is handed, and the `InputGraphSpec` that decides it.
* `model` -- the contract itself, and `integrate_acceleration`, the helper that
  turns a predicted acceleration into a frame in the raw schema.
"""

from .box import Box
from .graph import (
    EDGE_MODES,
    INPUT_GRAPH_SCHEMA,
    VELOCITY_MODES,
    InputGraphSpec,
)
from .model import (
    SimulatorModel,
    current_velocity,
    integrate_acceleration,
    predicted_acceleration,
)
from .raw import (
    ALWAYS_STORED,
    DERIVABLE_FIELDS,
    FieldSpec,
    FrameSchema,
    describe_schema,
    validate_raw_frame,
)

__all__ = [
    # the model contract
    "SimulatorModel",
    "integrate_acceleration",
    "predicted_acceleration",
    "current_velocity",
    # the input graph
    "InputGraphSpec",
    "INPUT_GRAPH_SCHEMA",
    "VELOCITY_MODES",
    "EDGE_MODES",
    # the raw schema
    "FrameSchema",
    "FieldSpec",
    "ALWAYS_STORED",
    "DERIVABLE_FIELDS",
    "Box",
    "describe_schema",
    "validate_raw_frame",
]
