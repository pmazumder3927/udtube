"""Collators.

* TextCollator is for raw text data only.
* ConlluCollator is for text data from CoNLL-U files; it doesn't matter whether
  that data has labels or not, but it can handle padding tensors if this is a
  labeled batch.
"""

from typing import List

import spacy
import transformers

from . import batches, datasets, indexes, padding


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


class TextCollator:
    """Collator for raw text data.

    Args:
        pretokenizer: UDPipe-style tokenizer.
        tokenizer: transformer autotokenizer.
    """

    def __init__(
        self,
        pretokenizer: spacy.language.Language,
        tokenizer: transformers.AutoTokenizer,
    ):
        self.pretokenizer = pretokenizer
        self.tokenizer = Tokenizer(tokenizer)

    def __call__(self, texts: List[str]) -> batches.TextBatch:
        pretokens = [
            [token.text for token in self.pretokenizer(text)] for text in texts
        ]
        tokens = self.tokenizer(pretokens)
        return batches.TextBatch(texts, tokens)


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

    def __call__(self, itemlist: List[datasets.Item]) -> batches.ConlluBatch:
        return batches.ConlluBatch(
            texts=[item.text for item in itemlist],
            tokens=self.tokenizer([item.form for item in itemlist]),
            # Looks ugly, but this just pads and stacks data for whatever
            # classification tasks are enabled.
            upos=(
                padding.pad_tensors(
                    [item.upos for item in itemlist], self.index.upos.pad_idx
                )
                if self.index.has_upos
                else None
            ),
            xpos=(
                padding.pad_tensors(
                    [item.xpos for item in itemlist], self.index.xpos.pad_idx
                )
                if self.index.has_xpos
                else None
            ),
            lemma=(
                padding.pad_tensors(
                    [item.lemma for item in itemlist],
                    self.index.lemma.pad_idx,
                )
                if self.index.has_lemma
                else None
            ),
            feats=(
                padding.pad_tensors(
                    [item.feats for item in itemlist],
                    self.index.feats.pad_idx,
                )
                if self.index.has_feats
                else None
            ),
        )
