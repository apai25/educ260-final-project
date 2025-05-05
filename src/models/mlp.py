from typing import List

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dims: int,
        activation: str = "relu",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        elif activation == "gelu":
            act_fn = nn.GELU()

        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(act_fn)
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
