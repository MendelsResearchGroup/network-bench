"""Baseline models and the registry that makes a model benchmarkable."""

from .edge_mlp import EdgeMLP, DeltaEdgeMLP
from .attention_mlp import AttentionEdgeMLP
from .frozen import Frozen
from .gns import GNS
from .linear_floor import LinearFloor
from .mlp import NodeMLP
from .tiny_model import TinyVelocityMLP
from .registry import MODELS, build, defaults, keys, register

__all__ = ["MODELS", "build", "defaults", "keys", "register", "GNS", "NodeMLP", "Frozen", "LinearFloor", "TinyVelocityMLP", "EdgeMLP", "DeltaEdgeMLP", "AttentionEdgeMLP"]
