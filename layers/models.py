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


class Biaffine(nn.Module):
    """
    Credit to
    https://github.com/Unipisa/diaparser/blob/94902d84021256ea010fee1db2de9fed9bad1d76/diaparser/modules/affine.py#L7
    An implementation of a Biaffine layer based on Deep Biaffine Parsing (Duzat & Manning 2017)
    https://arxiv.org/abs/1611.01734.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        bias_x: bool = True,
        bias_y: bool = True,
    ):
        """
        Initializes the instance based on user input.

        Args:
            in_dim: the size of the input embedding dimension of the input
            out_dim: the output dimension of the MLP. Default is 1
            bias_x: if True, adds a dimension representing a bias term to the x tensor
            bias_y: if True, adds a dimension representing a bias term to the y tensor
        """
        super(Biaffine, self).__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.Tensor(out_dim, in_dim + bias_x, in_dim + bias_y)
        )
        # Kaiming normal makes sense as the activation function used later is leaky_relU
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(dim=1)
        return s
