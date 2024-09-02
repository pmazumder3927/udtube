"""Logits object."""

import torch
from torch import nn


class Logits(nn.Module):
    """Logits from the classifier forward pass.

    Each tensor is either null or of shape N x C x L."""

    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]

    def __init__(self, upos=None, xpos=None, lemma=None, feats=None):
        super().__init__()
        self.register_buffer("upos", upos)
        self.register_buffer("xpos", xpos)
        self.register_buffer("lemma", lemma)
        self.register_buffer("feats", feats)

    @property
    def has_upos(self) -> bool:
        return self.upos is not None

    @property
    def has_xpos(self) -> bool:
        return self.xpos is not None

    @property
    def has_lemma(self) -> bool:
        return self.lemma is not None

    @property
    def has_feats(self) -> bool:
        return self.feats is not None
