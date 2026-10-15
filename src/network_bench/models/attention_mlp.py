"""One attention-pooled context vector shared by the nodes of one trajectory."""

import torch
from torch import nn

from .edge_mlp import DeltaEdgeMLP


class GlobalContext(nn.Module):
    """Pool node features into a small context, then add it back to every node."""

    def __init__(self, width: int, context_dim: int):
        super().__init__()
        self.score = nn.Linear(width, 1, bias=False)
        self.value = nn.Linear(width, context_dim)
        self.output = nn.Linear(context_dim, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, nodes):
        weights = torch.softmax(self.score(nodes), dim=0)
        context = (weights * self.value(nodes)).sum(dim=0, keepdim=True)
        return nodes + self.output(context)


class AttentionEdgeMLP(DeltaEdgeMLP):
    """Velocity-difference edge MLP with one global attention head before its MLP.

    The benchmark passes one system at a time: pooling never mixes trajectories.
    Attention costs O(nodes), with no pairwise attention matrix or message rounds.
    """

    def __init__(self, spec, target_scale, *, hidden_dim: int = 128, depth: int = 4,
                 context_dim: int = 32):
        super().__init__(spec, target_scale, hidden_dim=hidden_dim, depth=depth)
        self.hyperparameters["context_dim"] = context_dim
        self.node_network = nn.Sequential(
            GlobalContext(spec.node_feature_width + hidden_dim, context_dim),
            self.node_network,
        )
