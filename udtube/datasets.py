"""Datasets."""

import collections
import json
import logging
import os
import re

from typing import List, Dict, Tuple, Union, TextIO, Iterator

import conllu
import joblib
from sklearn import preprocessing
from torch.utils import data

from . import defaults, edit_scripts


class FileNameError(Exception):
    pass


class ConlluMapDataset(data.Dataset):
    """CoNLL-U data set.

    This class loads the entire file into memory and is therefore only
    suitable for smaller data sets.

    Args:
        conllu_file: path to the CoNLL-U file.
        reverse_edits: if True, uses reverse edits (e.g., for suffixal
            languages).
        model_dir: model directory.
        train: if True, the label encoders are fit.
    """

    # From https://universaldependencies.org/u/pos/index.html.

    UPOS_CLASSES = [
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

    # TODO: rename `model_dir` and `reverse_edits` variables.
    # TODO: move documentation to the class level.

    def __init__(
        self,
        conllu_file: str,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        model_dir: str = "UDTube",
        train: bool = False,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.all_words = []
        self.conllu_file = conllu_file
        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        # Sets up label encoders.
        if conllu_file:
            self.xpos_classes = self._get_all_classes("xpos")
            self.feats_classes = self._get_all_classes("feats")
            self.lemma_classes = self._get_all_classes("lemma")
            if train:
                self.upos_encoder = preprocessing.LabelEncoder()
                self.xpos_encoder = preprocessing.LabelEncoder()
                self.ufeats_encoder = preprocessing.LabelEncoder()
                self.lemma_encoder = preprocessing.LabelEncoder()
                self._fit_label_encoders()
                self.multiword_table = {}
            else:
                self._load_label_encoders()
            self.data_set = self._get_data(train)
        else:
            self.data_set = []

    def _fit_label_encoders(self) -> None:
        self.upos_encoder.fit(defaults.SPECIAL_TOKENS + self.UPOS_CLASSES)
        self.xpos_encoder.fit(defaults.SPECIAL_TOKENs + self.xpos_classes)
        self.ufeats_encoder.fit(defaults.SPECIAL_TOKENS + self.feats_classes)
        self.lemma_encoder.fit(defaults.SPECIAL_TOKENS + self.lemma_classes)
        try:
            os.mkdir(self.model_dir)
        except FileExistsError:
            pass
        logging.info(f"Now saving label encoders for {self.conllu_file}")
        joblib.dump(self.upos_encoder, f"{self.model_dir}/upos_encoder.joblib")
        joblib.dump(self.xpos_encoder, f"{self.model_dir}/xpos_encoder.joblib")
        joblib.dump(
            self.ufeats_encoder, f"{self.model_dir}/ufeats_encoder.joblib"
        )
        joblib.dump(
            self.lemma_encoder, f"{self.model_dir}/lemma_encoder.joblib"
        )

    def _load_label_encoders(self) -> None:
        self.lemma_encoder = joblib.load(
            f"{self.model_dir}/lemma_encoder.joblib"
        )
        self.ufeats_encoder = joblib.load(
            f"{self.model_dir}/ufeats_encoder.joblib"
        )
        self.upos_encoder = joblib.load(
            f"{self.model_dir}/upos_encoder.joblib"
        )
        self.xpos_encoder = joblib.load(
            f"{self.model_dir}/xpos_encoder.joblib"
        )

    # TODO: docs.
    # TODO: typing.

    def _get_all_classes(
        self, lname: str
    ) -> List[Union[str, None, int, List[int], Dict[str, str]]]:
        """Gets all cllasses observed in the training set."""
        classes = []
        with open(self.conllu_file) as source:
            for token_list in conllu.parse_incr(
                source, field_parsers=defaults.OVERRIDDEN_FIELD_PARSERS
            ):
                for token in token_list:
                    item = token[lname] if token[lname] else "_"
                    if lname != "lemma" and item not in classes:
                        classes.append(item)
                    elif lname == "lemma":
                        lrule = str(
                            self.e_script(token["form"].lower(), item.lower())
                        )
                        if lrule not in classes:
                            classes.append(lrule)
        return classes

    # TODO: docs.

    def _get_data(self, train: bool) -> List[Tuple[int]]:
        """Loads in CoNLL-U data."""
        data = []
        with open(self.conllu_file) as source:
            for token_list in conllu.parse_incr(
                source, field_parsers=defaults.OVERRIDDEN_FIELD_PARSERS
            ):
                sentence = token_list.metadata["text"]
                uposes = []
                xposes = []
                lemma_rules = []
                ufeats = []
                for i, token in enumerate(token_list):
                    if isinstance(token["id"], tuple):
                        if train:
                            # This is a multi-word token.
                            # These are not perfect indices because of
                            # multiword tokens.
                            start, sep, end = token["id"]
                            if not sep == ".":
                                # When sep is ., this is an elided token.
                                x = (end - start) + i
                                # This is i + 1 is because we want the next
                                # token.
                                words = [
                                    t["form"]
                                    for t in token_list[i + 1 : x + 2]
                                ]
                                if token["form"] in self.multiword_table:
                                    self.multiword_table[token["form"]][
                                        "frequency"
                                    ] += 1
                                else:
                                    self.multiword_table[token["form"]] = {
                                        "frequency": 1,
                                        "words": words,
                                    }
                        # We don't want to add it to the data set.
                        continue
                    l_rule = str(
                        self.e_script(
                            token["form"].lower(), token["lemma"].lower()
                        )
                    )
                    # Label encoders can't handle None.
                    if token["feats"] is None:
                        token["feats"] = "_"
                    # We have to check if the label in the data set is unknown.
                    # TODO: do something about the variable names here.
                    upos_ = (
                        token["upos"]
                        if token["upos"] in self.upos_encoder.classes_
                        else defaults.UNK_TAG
                    )
                    xpos_ = (
                        token["xpos"]
                        if token["xpos"] in self.xpos_encoder.classes_
                        else defaults.UNK_TAG
                    )
                    ufeat_ = (
                        token["feats"]
                        if token["feats"] in self.ufeats_encoder.classes_
                        else defaults.UNK_TAG
                    )
                    if l_rule not in self.lemma_encoder.classes_:
                        l_rule = defaults.UNK_TAG
                    uposes.append(upos_)
                    xposes.append(xpos_)
                    lemma_rules.append(l_rule)
                    ufeats.append(ufeat_)
                    self.all_words.append(token["form"].lower())
                uposes = self.upos_encoder.transform(uposes)
                xposes = self.xpos_encoder.transform(xposes)
                lemma_rules = self.lemma_encoder.transform(lemma_rules)
                ufeats = self.ufeats_encoder.transform(ufeats)
                data.append((sentence, uposes, xposes, lemma_rules, ufeats))
            if train:
                self.check_multiword_table()
                with open(
                    f"{self.model_dir}/multiword_dict.json", "w"
                ) as mw_tb:
                    json.dump(self.multiword_table, mw_tb, ensure_ascii=False)
        return data

    # TODO: docs.

    def check_multiword_table(self) -> None:
        """Computes multiword table.

        Multiword itmes that are more frequent as a single word are discarded.
        """
        single_cnt = collections.Counter(self.all_words)
        keep = {}
        for w, record in self.multiword_table.items():
            if w in single_cnt and single_cnt[w] > record["frequency"]:
                continue
            keep[w] = record
        self.multiword_table = keep

    # TODO: docs.

    def get_special_words(self) -> List[str]:
        """Computes list of non-standard words."""
        special_words = []
        with open(self.conllu_file) as source:
            for token_list in conllu.parse_incr(
                source, field_parsers=defaults.OVERRIDDEN_FIELD_PARSERS
            ):
                for token in token_list:
                    if isinstance(token["id"], int) and re.search(
                        r"\w+[.'-\\/%]|[.'-\\/%]\w+", token["form"]
                    ):
                        special_words.append(token["form"])
        return special_words

    def __len__(self) -> int:
        return len(self.data_set)

    # TODO: typing.

    def __getitem__(self, idx: int):
        return self.data_set[idx]


class ConlluIterDataset(data.IterableDataset):
    """Iterable CoNLL-U data set.

    This class parses the CoNLL-U files incrementally.

    Args:
        conllu_file: path to the CoNLL-U file.
    """

    def __init__(self, conllu_file: str):
        """Initializes the instance based on user input."""
        super().__init__()
        self.conllu_file = conllu_file

    def __iter__(self) -> Iterator[str]:
        if isinstance(self.conllu_file, str):
            self.conllu_file = open(self.conllu_file)
        return (
            token_list.metadata["text"]
            for token_list in conllu.parse_incr(
                self.conllu_file,
                field_parsers=defaults.OVERRIDDEN_FIELD_PARSERS,
            )
        )

    def __del__(self) -> None:
        if not isinstance(self.conllu_file, str):
            self.conllu_file.close()


class TextIterDataset(data.IterableDataset):
    """Iterable data set for text file data.

    This class is ideal for inference because it does not load the whole data
    set into memory.

    Args:
        text_file: The path to the text file.
    """

    def __init__(self, text_file: str):
        super().__init__()
        self.tf = text_file

    def __iter__(self) -> TextIO:
        if isinstance(self.tf, str):
            self.tf = open(self.tf)
        return self.tf

    def __del__(self) -> None:
        if hasattr(self, "tf") and self.tf:
            self.tf.close()
