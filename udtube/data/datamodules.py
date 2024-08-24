"""Data modules."""

import os
from typing import Optional

import lightning
import transformers
from torch.utils import data

from .. import defaults
from . import collators, datasets, indexes, parsers, pretokenizers, tokenizers


class Error(Exception):
    pass


class DataModule(lightning.LightningDataModule):
    """CoNLL-U data module.

    This class is initialized by the LightningCLI interface. It manages all
    data loading steps for UDTube. Training and validation are loaded into
    fully into memory; prediction data is loaded incrementally to avoid memory
    errors with large datasets.

    Args:
        model_dir: path for checkpoints, indexes, and logs.
        train: path to training CoNLL-U file.
        val: path to validation CoNLL-U file.
        predict: path to the prediction dataset; may either be text
            or CoNLL-U format (if the path ends with .conllu)
        test: path to the test CoNLL-U dataset.
        language: ISO-639-1 language code (for the pretokenizer).
        encoder: name of encoder on Hugging Face (for the tokenizer).
        reverse_edits: if true, uses reverse (suffixal) edit scripts.
        batch_size: batch size.
        index: an optional pre-computed index.
    """

    # TODO: argument documentation is incomplete.

    # TODO(#20): reconsider inference for CoNLL-U vs. text file.

    train: Optional[str]
    val: Optional[str]
    predict: Optional[str]
    test: Optional[str]

    parser: parsers.ConlluParser
    batch_size: int
    index: indexes.Index
    tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        train=None,
        val=None,
        predict=None,
        test=None,
        # Modeling options.
        language: str = defaults.LANGUAGE,
        encoder: str = defaults.ENCODER,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
        index: Optional[indexes.Index] = None,
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
        # FIXME I need to be able to load the index from a file too.
        self.index = index if index else self._make_index(model_dir)
        # TODO: this is not needed if all files are CoNLL-U format. Can we
        # hold off loading it until we know we need it?
        self.pretokenizer = pretokenizers.load(language)
        self.tokenizer = tokenizers.load(encoder)

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

    def _make_index(self, model_dir: str) -> indexes.Index:
        # We don't need to collect the upos vocabulary because "u" stands for
        # "universal" here.
        xpos_vocabulary = set() if self.has_xpos else None
        lemma_vocabulary = set() if self.has_lemma else None
        feats_vocabulary = set() if self.has_feats else None
        for _, _, _, xpos, lemma, feats in self.parser.samples(self.train):
            if self.has_xpos:
                xpos_vocabulary.update(xpos)
            if self.has_lemma:
                lemma_vocabulary.update(lemma)
            if self.has_feats:
                feats_vocabulary.update(feats)
        result = indexes.Index(
            upos=(
                indexes.Vocabulary(self.UPOS_VOCABULARY)
                if self.has_upos
                else None
            ),
            xpos=(
                indexes.Vocabulary(xpos_vocabulary)
                if xpos_vocabulary
                else None
            ),
            lemma=(
                indexes.Vocabulary(lemma_vocabulary)
                if lemma_vocabulary
                else None
            ),
            feats=(
                indexes.Vocabulary(feats_vocabulary)
                if feats_vocabulary
                else None
            ),
        )
        # Writes it to the model directory.
        try:
            os.mkdir(model_dir)
        except FileExistsError:
            pass
        result.write(model_dir)
        return result

    # Properties.

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
        return len(self.index.upos) if self.has_upos else 0

    @property
    def xpos_tagset_size(self) -> int:
        return len(self.index.xpos) if self.has_xpos else 0

    @property
    def lemma_tagset_size(self) -> int:
        return len(self.index.lemma) if self.has_lemma else 0

    @property
    def feats_tagset_size(self) -> int:
        return len(self.index.feats) if self.has_feats else 0

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            datasets.ConlluMapDataset(
                list(self.parser.samples(self.train)),
                self.index,
                self.parser,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.index,
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
                self.index,
                self.parser,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.index,
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
                    self.index,
                ),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )
        else:
            return data.DataLoader(
                datasets.TextIterDataset(self.parser),
                collate_fn=collators.TextCollator(
                    self.pretokenizer,
                    self.tokenizer,
                ),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            datasets.ConlluMapDataset(
                list(self.parser.samples(self.test)),
                self.index,
                self.parser,
            ),
            collate_fn=collators.ConlluCollator(
                self.tokenizer,
                self.index,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
