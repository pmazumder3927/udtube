"""Datasets and related utilities.

* TextIterDataset is a text dataset, loaded incrementally.
* ConlluIterDataset is a CoNLL-U dataset (labeled or not, it doedsn't matter),
  loaded incrementally.
* ConlluMapDataset is a labeled dataset, loaded greedily.

It's easy to imagine a "TextMapDataset" too, but it wouldn't serve any
particular purpose."""

import dataclasses
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    TextIO,
)

import torch
from torch import nn
from torch.utils import data

from . import indexes, parsers


class Item(nn.Module):
    """Tensors representing a single sentence."""

    text: str
    form: List[str]
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]

    def __init__(
        self, text, form, upos=None, xpos=None, lemma=None, feats=None
    ):
        super().__init__()
        self.text = text
        self.form = form
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


@dataclasses.dataclass
class TextIterDataset(data.IterableDataset):
    """Iterable data set for text file data.

    This class can be used for inference over large files because it does not
    load the whole data set into memory.

    Args:
        text_file: path to the input text file.
    """

    text_file: str

    def __iter__(self) -> TextIO:
        with open(self.text_file, "r") as source:
            for line in source:
                yield line.rstrip()


@dataclasses.dataclass
class ConlluIterDataset(data.IterableDataset):
    """Iterable CoNLL-U data set.

    This class can be used for inference over large files because it does not
    load the whole data set into memory.

    CoNLL-U fields other than `text` are simply ignored.

    Args:
        conllu_file: path to the input CoNLL-U file.
    """

    conllu_file: str

    def __iter__(self) -> Iterator[str]:
        for tokenlist in parsers.tokenlist_iterator(self.conllu_file):
            yield tokenlist.metadata["text"]


@dataclasses.dataclass
class ConlluMapDataset(data.Dataset):
    """Mappable CoNLL-U data set.

    This class loads the entire file into memory and is therefore only
    suitable for smaller data sets.

    It is mostly used during training and testing.
    """

    samples: List[parsers.SampleType]
    index: indexes.Index  # Usually copied from the DataModule.
    parser: parsers.ConlluParser  # Ditto.

    @property
    def has_upos(self) -> bool:
        return self.parser.use_upos

    @property
    def has_xpos(self) -> bool:
        return self.parser.use_xpos

    @property
    def has_lemma(self) -> bool:
        return self.parser.use_lemma

    @property
    def has_feats(self) -> bool:
        return self.parser.use_feats

    # Encoding.

    def _encode(
        self, labels: Iterable[str], index: indexes.Index
    ) -> torch.Tensor:
        """Encodes a tensor.

        Args:
            labels: iterable of labels.
            index: an index.

        Returns:
            Tensor of encoded labels.
        """
        return torch.tensor(
            [index(label) for label in labels], dtype=torch.long
        )

    def encode_upos(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes universal POS tags.

        Args:
            labels: iterable of universal POS strings.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.indexes.upos)

    def encode_xpos(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes language-specific POS tags.

        Args:
            labels: iterable of label-specific POS strings.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.indexes.xpos)

    def encode_lemma(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes lemma (i.e., edit script) tags.

        Note that there's no processing of the lemmatization rule, which is
        assumed to already be a string.

        Args:
            labels: iterable of lemma tags.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.indexes.lemma)

    def encode_feats(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes morphological feature tags.

        Args:
            labels: iterable of feature tags.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.indexes.feats)

    # Decoding.

    def _decode(
        self, indices: torch.Tensor, index: indexes.Index
    ) -> Iterator[List[str]]:
        """Decodes a tensor.

        Args:
            indices: 2d tensor of indices.
            index: index.

        Yields:
            List[str]: Lists of decoded strings.
        """
        for idx in indices.cpu().numpy():
            yield [
                index.get_symbol(c) for c in idx if c not in index.special_idx
            ]

    def decode_upos(self, indices: torch.Tensor) -> Iterator[List[str]]:
        """Decodes an upos tensor.

        Args:
            indices: 2d tensor of indices.

        Yields:
            List[str]: Decoded strings.
        """
        return self._decode(indices, self.indexes.upos)

    def decode_xpos(self, indices: torch.Tensor) -> Iterator[List[str]]:
        """Decodes an xpos tensor.

        Args:
            indices: 2d tensor of indices.

        Yields:
            List[str]: Decoded strings.
        """
        return self._decode(indices, self.indexes.xpos)

    # Does this need to be rewired to get the form so it can lemmatize?

    def decode_lemma(self, indices: torch.Tensor) -> Iterator[List[str]]:
        """Decodes a lemma tensor.

        Note that there's no processing of the lemmatization rule, which is
        assumed to already be a string.

        Args:
            indices: 2d tensor of indices.

        Yields:
            List[str]: Decoded strings.
        """
        return self._decode(indices, self.indexes.lemma)

    # Required API.

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx: index.

        Returns:
            Item.
        """
        text, form, upos, xpos, lemma, feats = self.samples[idx]
        return Item(
            text=text,
            form=form,
            upos=self.encode_upos(upos) if self.has_upos else None,
            xpos=self.encode_xpos(xpos) if self.has_upos else None,
            lemma=self.encode_lemma(lemma) if self.has_lemma else None,
            feats=self.encode_feats(feats) if self.has_feats else None,
        )
