import torch
from torch import Tensor, nn
import torch.nn.functional as F
import sys
import math


class ClassicMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, hidden_size),
            )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.dropout(self.norm(self.mlp(x)))