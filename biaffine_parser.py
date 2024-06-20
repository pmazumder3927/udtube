from typing import Tuple

import lightning.pytorch as pl
import torch

from layers.models import MLP, Biaffine


class BiaffineParser(pl.LightningModule):
    """Biaffine Dependency Parser based on  Deep Biaffine Parsing (Dozat & Manning 2017)
    https://arxiv.org/abs/1611.01734..
    Credit https://github.com/daandouwe/biaffine-dependency-parser/tree/9338c6fde6de5393ac1bbdd6a8bb152c2a015a6c
    """

    def __init__(
        self,
        input_size: int,
        dropout: float,
        num_labels: int,
        arc_hidden: int = 400,
        label_hidden: int = 100,
    ):
        """
        Creates a BiaffineParser object.

        Args:
            input_size: the size of the input embedding dimension
            dropout: the amount of dropout for the model
            num_labels: the amount of possible labels
            arc_hidden: the dimension to reduce input size to for arc classification
            label_hidden: the dimesnion to reduce input size to for the label classification
        """
        super().__init__()

        # Arc MLPs
        self.arc_mlp_h = MLP(input_size, arc_hidden, dropout=dropout)
        self.arc_mlp_d = MLP(input_size, arc_hidden, dropout=dropout)

        # Label MLPs
        self.lab_mlp_h = MLP(input_size, label_hidden, dropout=dropout)
        self.lab_mlp_d = MLP(input_size, label_hidden, dropout=dropout)

        # BiAffine layers
        self.arc_biaffine = Biaffine(arc_hidden, 1, bias_y=False)
        self.lab_biaffine = Biaffine(label_hidden, num_labels)

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the score matrices for the arcs and labels."""
        arc_h = self.arc_mlp_h(batch)
        arc_d = self.arc_mlp_d(batch)
        lab_h = self.lab_mlp_h(batch)
        lab_d = self.lab_mlp_d(batch)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)

        return S_arc, S_lab
