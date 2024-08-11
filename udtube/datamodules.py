"""Data modules."""

import functools
import json
import re

from typing import Callable, Optional

import pytorch_lightning as pl
import spacy_udpipe
import transformers
import torch
from torch.utils import data

from . import batches, datasets, defaults


# TODO: will it break if we inherit from tokenizers.Encoding?
class CustomEncoding:
    """Encoding object emulating tokenizer encodings object.

    For models that are not built on rust (like ByT5) this object is needed to
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

    # TODO: rename `path_name`, `lang_with_space`, and
    # `reverse_edits` variables.

    # TODO(#20): reconsider inference for CoNLL-U vs. text file.

    def __init__(
        self,
        path_name: str = "UDTube",
        encoder: str = defaults.ENCODER,
        language: str = defaults.LANGUAGE,
        train_dataset: Optional[str] = None,
        val_dataset: Optional[str] = None,
        predict_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None,
        batch_size: int = defaults.BATCH_SIZE,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        lang_with_space: bool = defaults.LANG_WITH_SPACE,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        # TODO: only catch relevant exceptions and remove the "noqa".
        try:
            self.pretokenizer = spacy_udpipe.load(language)
        except:  # noqa: E722
            spacy_udpipe.download(language)
            self.pretokenizer = spacy_udpipe.load(language)
        self.reverse_edits = reverse_edits
        self.lang_with_space = lang_with_space
        self.train_dataset = datasets.ConlluMapDataset(
            train_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=True,
        )
        self.val_dataset = datasets.ConlluMapDataset(
            val_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=False,
        )
        if predict_dataset and predict_dataset.endswith(".conllu"):
            self.predict_dataset = datasets.ConlluIterDataset(predict_dataset)
        else:
            self.predict_dataset = datasets.TextIterDataset(predict_dataset)
        self.test_dataset = datasets.ConlluMapDataset(
            test_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=False,
        )
        with open(f"{path_name}/multiword_dict.json", "r") as mw_tb:
            self.multiword_table = json.load(mw_tb)
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder)
        self.checkpoint = checkpoint
        if self.train_dataset:
            # this is a bit hacky, but not sure how to do this with setup & CLI
            # + 3 is the padding & unk tok & _
            self.pos_classes_count = len(defaults.SPECIAL_TOKENS) + len(
                self.train_dataset.UPOS_CLASSES
            )
            self.xpos_classes_count = len(defaults.SPECIAL_TOKENS) + len(
                self.train_dataset.xpos_classes
            )
            self.lemma_classes_count = len(defaults.SPECIAL_TOKENS) + len(
                self.train_dataset.lemma_classes
            )
            self.feats_classes_count = len(defaults.SPECIAL_TOKENS) + len(
                self.train_dataset.feats_classes
            )
        else:
            self._set_values_from_path_name()

    def _set_values_from_path_name(self) -> None:
        assert self.checkpoint, "no model checkpoint found"
        hps = torch.load(self.checkpoint)["hyper_parameters"]
        self.pos_classes_count = hps.get("pos_out_label_size", 0)
        self.xpos_classes_count = hps.get("xpos_out_label_size", 0)
        self.lemma_classes_count = hps.get("lemma_out_label_size", 0)
        self.feats_classes_count = hps.get("feats_out_label_size", 0)

    # TODO: typing.

    def _adjust_sentence(self, sentences, lang_with_space=True):
        delimiter = " " if lang_with_space else ""
        new_sents = []
        # We need to keep track of these to write out multiword tokens
        # preprocessing sentences. The table is usually small so this is not
        # too expensive.
        replacements = []
        for sentence in sentences:
            new_sentence = sentence
            rep_i = []
            for k, v in self.multiword_table.items():
                if re.search(rf"\b{k}\b", sentence) and v["frequency"] > 2:
                    new_sentence = re.sub(
                        rf"\b{k}\b", delimiter.join(v["words"]), new_sentence
                    )
                    rep_i.append((k, delimiter.join(v["words"])))
            new_sents.append(new_sentence)
            replacements.append(rep_i)
        return new_sents, replacements

    # TODO: typing.

    def _pretokenize(self, sentences):
        batch_toks = []
        for sentence in sentences:
            tokens = [token.text for token in self.pretokenizer(sentence)]
            batch_toks.append(tokens)
        return batch_toks

    # TODO: more precise typing.

    def _collator(self, conllu=True) -> Callable:
        preprocessing = (
            self.conllu_preprocessing
            if conllu
            else self.txt_file_preprocessing
        )
        return functools.partial(
            preprocessing,
            tokenizer=self.tokenizer,
            lang_with_space=self.lang_with_space,
        )

    # TODO: typing.
    # TODO: docs.

    def conllu_preprocessing(
        self,
        batch,
        tokenizer: transformers.AutoTokenizer,
        lang_with_space=True,
    ):
        sentences, *args = zip(*batch)
        sentences, replacements = self._adjust_sentence(
            sentences, lang_with_space=lang_with_space
        )
        pretoks = self._pretokenize(sentences)
        # We load in tokens by batch to avoid using the inefficient max_length
        # padding.
        tokenized_x = tokenizer(
            pretoks,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
        )
        if not tokenized_x.encodings:
            # This is necessary for ByT5, which doesn't seem to have a fast
            # tokenizer.
            # TODO: variable names.
            encodings_ = []
            for sent in pretoks:
                idxs = []
                toks = []
                for i, word in enumerate(sent):
                    # TODO: variable names.
                    toks_ = list(word.encode("utf-8"))
                    num_toks = len(toks_)
                    idxs.extend([i] * num_toks)
                    toks.extend(toks_)
                encodings_.append(CustomEncoding(word_ids=idxs, tokens=toks))
            tokenized_x.custom_encodings = encodings_
        return batches.ConlluBatch(tokenized_x, sentences, replacements, *args)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=False),
        )

    # TODO: typing.
    # TODO: docs.

    def txt_file_preprocessing(
        self,
        sentences,
        tokenizer: transformers.AutoTokenizer,
        lang_with_space=True,
    ):
        sentences, replacements = self._adjust_sentence(
            sentences, lang_with_space=lang_with_space
        )
        pretoks = self._pretokenize(sentences)
        tokenized_x = tokenizer(
            pretoks,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
        )
        return batches.TextBatch(tokenized_x, sentences, replacements)

    # TODO: is this necessary or wise?

    def teardown(self, stage: str) -> None:
        if stage == "predict":
            del self.predict_dataset
