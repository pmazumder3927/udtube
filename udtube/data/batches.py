"""Batch objects.

This is closely based on the implementation in Yoyodyne:

    https://github.com/CUNY-CL/yoyodyne/blob/master/yoyodyne/data/indexes.py

However, there are some different concerns here; for instance padding is much
simpler.

* TextBatch contains tokenized data only.
* ConlluBatch contains tokenized and labeled data.
"""

from typing import List, Optional, Union

import torch
import transformers
from torch import nn


class Batch(nn.Module):
    """Base class for batches"""

    texts: List[str]
    tokens: transformers.BatchEncoding

    def __init__(self, texts, tokens):
        super().__init__()
        self.texts = texts
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.texts)

    def to(self, device: Union[str, torch.device, int]) -> "Batch":
        # `tokens` is a dictionary, so it can't be registered as a module
        # buffer, but it itself supports a version of `.to`.
        self.tokens = self.tokens.to(device)
        return super().to(device)


class TextBatch(nn.Module):
    """Text data batch.

    Args:
        tokens: the encoded batch, the output of the tokenizer.
        sentences: list of sentences.
    """

    pass


class ConlluBatch(Batch):
    """CoNLL-U data batch.

    This can handle padded label tensors if present.

    Args:
        tokens: the encoded batch, the output of the tokenizer.
        sentences: list of sentences.
        pos: optional padded tensor of POS labels.
        xpos: optional padded tensor of language-specific POS labels.
        lemma: optional padded tensor of lemma labels.
        feats: optional padded tensor of feats labels.
    """

    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]

    def __init__(
        self,
        texts,
        tokens,
        upos=None,
        xpos=None,
        lemma=None,
        feats=None,
    ):
        super().__init__(texts, tokens)
        # None is harmless here.
        self.register_buffer("upos", upos)
        self.register_buffer("xpos", xpos)
        self.register_buffer("lemma", lemma)
        self.register_buffer("feats", feats)
