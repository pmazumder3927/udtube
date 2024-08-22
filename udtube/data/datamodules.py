"""Data modules."""

from typing import Optional

import pytorch_lightning as pl
import transformers
from torch.utils import data

from . import collators, datasets, indexes, parsers
from .. import defaults


class Error(Exception):
    pass


# TODO: will it break if we inherit from tokenizers.Encoding?
class CustomEncoding:
    """Encoding object emulating tokenizer encodings object.

    For models that are not built on Rust (like ByT5) this object is needed to
    preserve the implementation of word_to_tokens.
    """

    # TODO: typing.

    def __init__(self, word_ids, tokens):
        self.word_ids = word_ids
        self.tokens = tokens

    # TODO: typing.

    def word_to_tokens(self, word_id):
        # TODO: less naive approach.
        start = self.word_ids.index(word_id)
        stop = start + 1
        for i, idx in enumerate(self.word_ids[start + 1 :]):
            if idx == word_id:
                stop += 1
            else:
                break
        return start, stop


class DataModule(pl.LightningDataModule):
    """CoNLL-U data module.

    This class is initialized by the LightningCLI interface. It manages all
    data loading steps for UDTube. Training and validation are loaded into
    fully into memory; prediction data is loaded incrementally to avoid memory
    errors with large datasets.

    Args:
        path_name: directory for model assets.
        encoder: name of encoder on Hugging Face.
        train_dataset: path to training CoNLL-U file.
        val_dataset: path to validation CoNLL-U file.
        predict_dataset: path to the prediction dataset; may either be text
            or CoNLL-U format (if the path ends with .conllu)
        batch_size: batch size.
        reverse_edits: if true, uses reverse (suffixal) edit scripts.
        checkpoint: path to the model checkpoint.
    """

    # TODO: argument documentation is incomplete.

    # TODO(#20): reconsider inference for CoNLL-U vs. text file.

    train: Optional[str]
    val: Optional[str]
    predict: Optional[str]
    test: Optional[str]

    parser: parsers.ConlluParser
    batch_size: int
    indexes: indexes.Indexes
    tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        # Paths.
        *,
        train: Optional[str] = None,
        val: Optional[str] = None,
        predict: Optional[str] = None,
        test: Optional[str] = None,
        # Modeling options.
        encoder: str = defaults.ENCODER,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        # Collator options.
        batch_size: int = defaults.BATCH_SIZE,
        # Indexing.
        indexes: Optional[indexes.Indexes] = None,
    ):
        super().__init__()
        # Validates and stores data paths.
        if train:
            self._require_conllu_extension(train, "training")
            self.train = train
        if val:
            self._require_conllu_extension(train, "validation")
            self.val = val
        self.predict = predict
        if test:
            self._require_conllu_extension(test, "testing")
            self.test = test
        # Stores or computes other necessary parameters.
        self.parser = parsers.ConlluParser(
            use_upos=use_upos,
            use_xpos=use_xpos,
            use_lemma=use_lemma,
            use_feats=use_feats,
            reverse_edits=reverse_edits,
        )
        self.batch_size = batch_size
        # FIXME I need to be able to load the indexes from a file too.
        self.indexes = indexes if indexes else self._make_indexes()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder)

    @staticmethod
    def _require_conllu_extension(path: str, file_type: str) -> None:
        """Raises an exception if a path does not have a `.conllu` extension.

        Args:
            path: file path.
            file_type: type of file, used for the error message.

        Raises:
            Error: file must have a `.conllu` extension and be in CoNLL-U
                format.
        """
        if not path.endswith(".conllu"):
            raise Error(
                f"{file_type} file {path} must have a `.conllu` extension"
                "and be in CoNLL-U format"
            )

    # Index creation.

    # Based on: https://universaldependencies.org/u/pos/index.html.

    UPOS_VOCABULARY = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]

    def _make_index(self) -> indexes.Indexes:
        # We don't need to collect the upos vocabulary because it's universal.
        xpos_vocabulary = set() if self.has_xpos else None
        lemma_vocabulary = set() if self.has_lemma else None
        feats_vocabulary = set() if self.has_feats else None
        for _, _, xpos, lemma, feats in self.parser.samples(self.train):
            if self.has_xpos:
                xpos_vocabulary.update(xpos)
            if self.has_lemma:
                lemma_vocabulary.update(lemma)
            if self.has_feats:
                feats_vocabulary.update(feats)
        return indexes.Indexes(
            upos=self.UPOS_VOCABULARY if self.has_upos else None,
            xpos=xpos_vocabulary if xpos_vocabulary else None,
            lemma=lemma_vocabulary if lemma_vocabulary else None,
            feats=feats_vocabulary if feats_vocabulary else None,
        )

    def write_indexes(self, model_dir: str) -> None:
        """Writes the indexes to a file."""
        self.indexes.write(model_dir)

    # Metadata about the task, etc.

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

    @property
    def upos_tagset_size(self) -> int:
        return len(self.indexes.upos) if self.has_upos else 0

    @property
    def xpos_tagset_size(self) -> int:
        return len(self.indexes.xpos) if self.has_xpos else 0

    @property
    def lemma_tagset_size(self) -> int:
        return len(self.indexes.lemma) if self.has_lemma else 0

    @property
    def feats_tagset_size(self) -> int:
        return len(self.indexes.feats) if self.has_feats else 0

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            datasets.ConlluMapDataset(
                list(self.parser.samples(self.train)),
                self.indexes,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.indexes,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no val path"
        return data.DataLoader(
            datasets.ConlluMapDataset(
                list(self.parser.samples(self.val)),
                self.indexes,
                self.parser,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.indexes,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        # In this mode, either a .conllu file or a text file will do.
        if self.predict.endswith(".conllu"):
            return data.DataLoader(
                datasets.ConlluIterDataset(self.predict),
                collate_fn=collators.ConlluCollator(
                    self.tokenizer,
                    self.indexes,
                ),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )
        else:
            return data.DataLoader(
                datasets.TextIterDataset(self.parser),
                collate_fn=collators.TextCollator(self.tokenizer),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            datasets.ConlluMapDataset(
                list(self.parser.samples(self.test)),
                self.indexes,
                self.parser,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.indexes,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
