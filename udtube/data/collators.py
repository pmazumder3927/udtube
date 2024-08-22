"""Collators.

* TextCollator is for raw text data only.
* ConlluCollator is for text data from CoNLL-U files; it doesn't matter whether
  that data has labels or not, but it can handle padding tensors if this is a
  labeled batch.
"""

import dataclasses
from typing import List

import spacy
import transformers

from . import batches, datasets, indexes, padding


@dataclasses.dataclass
class TextCollator:
    """Collator for raw text data.

    Args:
        pretokenizer: UDPipe-style tokenizer.
        tokenizer: transformer autotokenizer.
    """

    pretokenizer: spacy.language.Language
    tokenizer: transformers.AutoTokenizer

    def __call__(self, texts: List[str]) -> batches.TextBatch:
        pretokens = [
            [token.text for token in self.pretokenize(text)] for text in texts
        ]
        tokens = self.tokenizer(
            pretokens,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )
        return batches.TextBatch(texts, tokens)


@dataclasses.dataclass
class ConlluCollator:
    """Collator for CoNLL-U data.

    It doesn't matter whether or not the data already has labels.

    Args:
        tokenizer: transformer autotokenizer.
        indexes: indexes.
    """

    tokenizer: transformers.AutoTokenizer
    indexes: indexes.Index

    def __call__(self, itemlist: List[datasets.Item]) -> batches.ConlluBatch:
        pretokens = [item.form for item in itemlist]
        tokens = self.tokenizer(
            pretokens,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )
        return batches.ConlluBatch(
            texts=[item.text for item in itemlist],
            tokens=tokens,
            # Looks ugly, but this just pads and stacks data for whatever
            # classification tasks are enabled.
            upos=(
                padding.pad_tensors(
                    [item.upos for item in itemlist], self.indexes.upos.pad_idx
                )
                if self.indexes.has_upos
                else None
            ),
            xpos=(
                padding.pad_tensors(
                    [item.xpos for item in itemlist], self.indexes.xpos.pad_idx
                )
                if self.indexes.has_xpos
                else None
            ),
            lemma=(
                padding.pad_tensors(
                    [item.lemma for item in itemlist],
                    self.indexes.lemma.pad_idx,
                )
                if self.indexes.has_lemma
                else None
            ),
            feats=(
                padding.pad_tensors(
                    [item.feats for item in itemlist],
                    self.indexes.feats.pad_idx,
                )
                if self.indexes.has_feats
                else None
            ),
        )
