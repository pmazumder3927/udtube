"""Symbol indexes.

This is closely based on the implementation in Yoyodyne:

    https://github.com/CUNY-CL/yoyodyne/blob/master/yoyodyne/data/indexes.py

The major difference here is that we have a separate index for each classifier
layer, since unlike in Yoyodyne, there is no such thing as a shared vocabulary
for the classifier layers.

Because of this, we also have the Indexes class, which holds instances of
the lower-level indexes, one for each enabled classifier head, and which
handles (de)serialization."""

import pickle
from typing import Dict, Iterable, List, Optional, Set

from .. import special


class Index:
    """Maintains an index over a vocabulary."""

    _index2symbol: List[str]
    _symbol2index: Dict[str, int]

    def __init__(self, vocabulary: Iterable[str]):
        # TODO: consider storing this in-class for logging purposes.
        self._index2symbol = special.SPECIALS + sorted(vocabulary)
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    # Lookup.

    def __call__(self, lookup: str) -> int:
        """Looks up index by symbol.

        Args:
            symbol (str).

        Returns:
            int.
        """
        return self._symbol2index.get(lookup, self.unk_idx)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index.

        Args:
            index (int).

        Returns:
            str.
        """
        return self._index2symbol[index]

    # Specials.

    @property
    def pad_idx(self) -> int:
        return self._symbol2index[special.PAD]

    @property
    def unk_idx(self) -> int:
        return self._symbol2index[special.UNK]

    @property
    def blank_idx(self) -> int:
        return self._symbol2index[special.BLANK]

    @property
    def special_idx(self) -> Set[int]:
        return {self.unk_idx, self.pad_idx, self.blank_idx}


class Indexes:
    """An index of indexes, with one index per enabled classification task.

    This also handles serialization and deserialization.

    Args:
        upos: optional index for universal POS tagging.
        xpos: optional index for language-specific POS tagging.
        lemma: optional index for lemmatization.
        feats: optional index for morphological tagging.
    """

    # TODO(#3): other things (multiword table?) can also be stashed here.

    upos: Optional[Index]
    xpos: Optional[Index]
    lemma: Optional[Index]
    feats: Optional[Index]

    def __init__(self, upos=None, xpos=None, lemma=None, feats=None):
        self.upos = Index(upos) if upos else None
        self.xpos = Index(xpos)

    # Properties.

    def has_upos(self) -> bool:
        return self.upos is not None

    def has_xpos(self) -> bool:
        return self.xpos is not None

    def has_lemma(self) -> bool:
        return self.lemma is not None

    def has_feats(self) -> bool:
        return self.feats is not None

    # Serialization.

    @staticmethod
    def path(model_dir: str) -> str:
        return f"{model_dir}/index.pkl"

    @classmethod
    def read(cls, model_dir: str) -> "Indexes":
        """Loads index.

        Args:
            model_dir (str).

        Returns:
            Index.
        """
        index = cls.__new__(cls)
        with open(cls.path(model_dir), "rb") as source:
            dictionary = pickle.load(source)
        for key, value in dictionary.items():
            setattr(index, key, value)
        return index

    def write(self, model_dir: str) -> None:
        with open(self.path(model_dir), "wb") as sink:
            pickle.dump(vars(self), sink)
