"""Collators."""

from typing import List

import torch
from torch import nn
import transformers

from . import batches, datasets, indexes


class Tokenizer:
    """Helper for tokenization.

    Args:
        tokenizer: transformer autotokenizer.
    """

    tokenizer: transformers.AutoTokenizer

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, pretokens: List[List[str]]
    ) -> transformers.BatchEncoding:
        return self.tokenizer(
            pretokens,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )


class ConlluCollator:
    """Collator for CoNLL-U data.

    It doesn't matter whether or not the data already has labels.

    Args:
        tokenizer: transformer autotokenizer.
        index: the index.
    """

    def __init__(
        self, tokenizer: transformers.AutoTokenizer, index: indexes.Index
    ):
        self.tokenizer = Tokenizer(tokenizer)
        self.index = index

    @staticmethod
    def pad_tensors(
        tensorlist: List[torch.Tensor],
        pad_idx: int = 0,
        dim=0,
    ) -> torch.Tensor:
        """Pads and stacks a list of tensors.

        Args:
            tensorlist: a list of tensors to be padded.
            pad_idx: padding index to use; defaults to 0.

        Returns:
            The padded and stacked tensor.
        """
        pad_max = max(len(tensor) for tensor in tensorlist)
        return torch.stack(
            [
                nn.functional.pad(
                    tensor, (0, pad_max - len(tensor)), value=pad_idx
                )
                for tensor in tensorlist
            ]
        )

    def __call__(self, itemlist: List[datasets.Item]) -> batches.ConlluBatch:
        return batches.ConlluBatch(
            texts=[item.text for item in itemlist],
            tokens=self.tokenizer([item.form for item in itemlist]),
            # Looks ugly, but this just pads and stacks data for whatever
            # classification tasks are enabled.
            upos=(
                self.pad_tensors(
                    [item.upos for item in itemlist], self.index.upos.pad_idx
                )
                if self.index.has_upos
                else None
            ),
            xpos=(
                self.pad_tensors(
                    [item.xpos for item in itemlist], self.index.xpos.pad_idx
                )
                if self.index.has_xpos
                else None
            ),
            lemma=(
                self.pad_tensors(
                    [item.lemma for item in itemlist],
                    self.index.lemma.pad_idx,
                )
                if self.index.has_lemma
                else None
            ),
            feats=(
                self.pad_tensors(
                    [item.feats for item in itemlist],
                    self.index.feats.pad_idx,
                )
                if self.index.has_feats
                else None
            ),
        )
