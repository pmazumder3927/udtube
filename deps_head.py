import lightning.pytorch as pl
import torch
from torch import nn

"""
Credit for a lot of this code goes to 
https://github.com/daandouwe/biaffine-dependency-parser/tree/9338c6fde6de5393ac1bbdd6a8bb152c2a015a6c
"""


class MLP(pl.LightningModule):
    def __init__(self, emb_size, output_dim, dropout=0.3):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(emb_size, output_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.MLP(batch)


class Biaffine(nn.Module):
    """https://github.com/Unipisa/diaparser/blob/master/diaparser/models/dependency.py#L189"""

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.Tensor(n_out, n_in + bias_x, n_in + bias_y)
        )
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)
        return s


class BiAffineParser(pl.LightningModule):
    """Biaffine Dependency Parser."""

    def __init__(
        self,
        input_size,
        dropout,
        num_arc_labels,
        arc_hidden=400,
        label_hidden=100,
    ):
        super().__init__()

        # Arc MLPs
        self.arc_mlp_h = MLP(input_size, arc_hidden, dropout=dropout)
        self.arc_mlp_d = MLP(input_size, arc_hidden, dropout=dropout)

        # Label MLPs
        self.lab_mlp_h = MLP(input_size, label_hidden, dropout=dropout)
        self.lab_mlp_d = MLP(input_size, label_hidden, dropout=dropout)

        # BiAffine layers
        self.arc_biaffine = Biaffine(arc_hidden, 1, bias_y=False)
        self.lab_biaffine = Biaffine(label_hidden, num_arc_labels)

        self.lab_ce_loss = nn.CrossEntropyLoss(ignore_index=num_arc_labels - 1)

    def forward(self, batch):
        """Compute the score matrices for the arcs and labels."""
        arc_h = self.arc_mlp_h(batch)
        arc_d = self.arc_mlp_d(batch)
        lab_h = self.lab_mlp_h(batch)
        lab_d = self.lab_mlp_d(batch)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)

        return S_arc, S_lab

    def lab_loss(self, S_lab, heads, labels, longest_sequence):
        """Compute the loss for the label predictions on the gold arcs (heads)."""
        heads = heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)  # [batch, n_labels, 1, sent_len]
        heads = torch.where(heads != longest_sequence, heads, 0)  # Replacing padding due to index error
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)  # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)  # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
        labels = labels.view(-1)  # [batch*sent_len]
        return self.lab_ce_loss(S_lab, labels)

    def loss(
        self,
        s_arc: torch.Tensor,
        s_rel: torch.Tensor,
        arcs: torch.Tensor,
        rels: torch.Tensor,
        longest_sequence,
            attn_masks
    ) -> torch.Tensor:
        r"""
        Computes the arc and tag loss for a sequence given gold heads and tags.

        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        # select the predicted relations towards the correct heads
        # have to drop a dim here (both are [batch_size, n_out, seq_len, seq_len])
        s_arc = s_arc * attn_masks.unsqueeze(dim=1)
        arc_loss = nn.functional.cross_entropy(s_arc, arcs, ignore_index=longest_sequence)
        rel_loss = self.lab_loss(s_rel, arcs, rels, longest_sequence)

        return arc_loss + rel_loss
