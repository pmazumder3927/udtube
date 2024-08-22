"""Batch objects.

This is closely based on the implementation in Yoyodyne:

    https://github.com/CUNY-CL/yoyodyne/blob/master/yoyodyne/data/indexes.py

However, there are some different concerns here; for instance padding is much
simpler.

* TextBatch contains tokenized data only.
* ConlluBatch contains tokenized and labeled data.
"""

import dataclasses
from typing import List, Optional

import transformers
import torch
from torch import nn


@dataclasses.dataclass
class Batch(nn.Module):
    """Base class for batch.

    Args:
        tokens: the encoded batch, the output of the tokenizer.
        sentences: list of sentences.
    """

    texts: List[str]
    tokens: transformers.BatchEncoding

    def __len__(self) -> int:
        return len(self.texts)


class TextBatch(Batch):
    """Text data batch.

    This consists of only tokenized inputs and sentences.
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
        pad_idx: int,
        upos=None,
        xpos=None,
        lemma=None,
        feats=None,
    ):
        super().__init__(tokens, texts)
        if upos:
            self.register_module("upos", upos)
        if xpos:
            self.register_module("xpos", xpos)
        if lemma:
            self.register_module("lemma", lemma)
        if feats:
            self.register_module("feats", feats)
