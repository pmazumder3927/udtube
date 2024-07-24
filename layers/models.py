import lightning.pytorch as pl
import torch
from torch import nn


class MLP(pl.LightningModule):
    """A simple multilayer perceptron class"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        """
        Initializes the instance based on user input. output_dim should probably be lower than emb_size

        Args
            in_dim: the size of the input embedding dimension of the input
            out_dim: the output dimension of the MLP
            dropout: dropout. Default is 0.3
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

    def forward(self, batch):
        return self.model(batch)
