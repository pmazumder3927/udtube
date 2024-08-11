"""Batch objects.

* `TextBatch` contains tokenized data only.
* `ConlluBatch` contains tokenized and labeled data

Both contain auxiliary sentence and replacement data as well.
"""

from typing import Iterable, List, Tuple

import transformers
from torch import nn


class Batch(nn.Module):
    """Base classe for batch.

    Args:
        tokens: the encoded batch, the output of the BERT tokenizer.
        sentences: list of sentences.
        replacements: replacements data; mapping of multi-word token
            replacements per sentence.
    """

    def __init__(
        self,
        tokens: transformers.BatchEncoding,
        sentences: Iterable[str],
        replacements: Iterable[Tuple[str, str]],
    ):
        super().__init__()
        self.tokens = tokens
        self.sentences = [s.rstrip("\n") for s in sentences]
        self.replacements = replacements

    def __len__(self) -> int:
        return len(list(self.sentences))


class TextBatch(Batch):
    """Text data batch.

    This consists of only tokenized inputs and sentences.
    """

    pass


class ConlluBatch(TextBatch):
    """CoNLL-U data batch.

    This also has labels and handles padding.

    Args:
        tokens: the encoded batch, the output of the BERT tokenizer.
        sentences: list of sentences.
        replacements: replacements data; mapping of multi-word token
            replacements per sentence.
        pos: tensor of POS labels.
        xpos: tensor of language-specific POS labels
        lemma: tensor of lemma labels.
        feats: tensor of feats labels.
    """

    # FIXME: the documentation does not match the typing here; it says
    # "tensor" but the the types say iterables of lists.

    def __init__(
        self,
        tokens: transformers.BatchEncoding,
        sentences: Iterable[str],
        replacements: Iterable[Tuple[str, str]],
        pos: Iterable[List[int]],
        xpos: Iterable[List[int]],
        lemma: Iterable[List[int]],
        feats: Iterable[List[int]],
    ):
        super().__init__(tokens, sentences, replacements)
        self.pos = pos
        self.xpos = xpos
        self.lemmas = lemma
        self.feats = feats
