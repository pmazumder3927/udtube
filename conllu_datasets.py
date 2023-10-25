import json
import os
import re
from collections import defaultdict, Counter

import conllu
import joblib
from sklearn.preprocessing import LabelEncoder
from torch import tensor
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from ud_compatibility import marry, languages
import numpy as np

import edit_scripts

# Overriding the parser for the conllu reader. This is needed so that feats can be read in as a str instead of a dict
OVERRIDDEN_FIELD_PARSERS = {
    "id": lambda line, i: conllu.parser.parse_id_value(line[i]),
    "xpos": lambda line, i: conllu.parser.parse_nullable_value(line[i]),
    "feats": lambda line, i: conllu.parser.parse_nullable_value(
        line[i]
    ),
    "head": lambda line, i: conllu.parser.parse_int_value(line[i]),
    "deps": lambda line, i: conllu.parser.parse_paired_list_value(line[i]),
    "misc": lambda line, i: conllu.parser.parse_dict_value(line[i]),
}


# TODO https://universaldependencies.org/u/feat/index.html if I need the full enumerable possibilities of ufeats.


class ConlluMapDataset(Dataset):
    """Conllu Map Dataset, used for small training/validation sets

    This class loads the entire dataset into memory and is therefore only suitable for smaller datasets
    """
    UNK_TAG = "[UNKNOWN]"
    PAD_TAG = "[PAD]"
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

    def __init__(self, conllu_file: str, reverse_edits: bool = False, path_name: str = "UDTube",
                 convert_to_um: bool = True, train: bool = False):
        """Initializes the instance based on user input.

        Args:
            conllu_file: The path to the conllu file to make the dataset from
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
            path_name: The dir where everything will get saved
            convert_to_um: Enable Universal Dependency (UD) file conversion to Universal Morphology (UM) format
            train: Should only be true for the train dataset. Used to decide if to fit encoders
        """
        super().__init__()
        self.path_name = path_name
        self.all_words = []
        self.conllu_file = self._convert_to_um(conllu_file, convert_to_um)
        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        # setting up label encoders
        if conllu_file:
            self.xpos_classes = self._get_all_classes("xpos")
            self.feats_classes = self._get_all_classes("feats")
            self.lemma_classes = self._get_all_classes("lemma")
            if train:
                self.upos_encoder = LabelEncoder()
                self.xpos_encoder = LabelEncoder()
                self.ufeats_encoder = LabelEncoder()
                self.lemma_encoder = LabelEncoder()
                self._fit_label_encoders()
                self.multiword_table = {}
            else:
                self._load_label_encoders()
            self.data_set = self._get_data(train)
        else:
            # Instantiation of empty class, happens in prediction
            self.data_set = []

    def _convert_to_um(self, conllu_file, convert_to_um):
        if not conllu_file:
            return
        # hopefully, as the marry.py repo stabilizes this will look less like an abomination
        if convert_to_um:
            language_match = re.search(r'^[a-zA-Z]+', conllu_file)
            if language_match:
                language = languages.get_lang(language_match.group(0))
                fc = marry.FileConverter(Path(conllu_file), language=language, clever=False)
                fc.convert() # writes a file
            if not language_match:
                raise Exception("File does not follow the naming convention for conllu files: <language>_<treebank>-<ud>-<split>.conllu")
            return conllu_file.replace('-ud-', '-um-')
        return conllu_file

    def _fit_label_encoders(self):
        self.upos_encoder.fit(self.UPOS_CLASSES + [self.PAD_TAG, self.UNK_TAG, "_"])
        self.xpos_encoder.fit(self.xpos_classes + [self.PAD_TAG, self.UNK_TAG, "_"])
        self.ufeats_encoder.fit(self.feats_classes + [self.PAD_TAG, self.UNK_TAG, "_"])
        self.lemma_encoder.fit(self.lemma_classes + [self.PAD_TAG, self.UNK_TAG, "_"])
        # saving all the encoders
        if not os.path.exists(self.path_name):
            os.mkdir(self.path_name)
        print(f"Now saving label encoders for {self.conllu_file}")
        joblib.dump(self.upos_encoder, f'{self.path_name}/upos_encoder.joblib')
        joblib.dump(self.xpos_encoder, f'{self.path_name}/xpos_encoder.joblib')
        joblib.dump(self.ufeats_encoder, f'{self.path_name}/ufeats_encoder.joblib')
        joblib.dump(self.lemma_encoder, f'{self.path_name}/lemma_encoder.joblib')

    def _load_label_encoders(self):
        self.lemma_encoder = joblib.load(f"{self.path_name}/lemma_encoder.joblib")
        self.ufeats_encoder = joblib.load(f"{self.path_name}/ufeats_encoder.joblib")
        self.upos_encoder = joblib.load(f"{self.path_name}/upos_encoder.joblib")
        self.xpos_encoder = joblib.load(f"{self.path_name}/xpos_encoder.joblib")

    def _get_all_classes(self, lname: str):
        """helper function to get all the classes observed in the training set"""
        classes = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tok in tk_list:
                    item = tok[lname] if tok[lname] else '_'
                    if lname != "lemma" and item not in classes:
                        classes.append(item)
                    elif lname == "lemma":
                        lrule = str(
                            self.e_script(
                                tok["form"].lower(), item.lower()
                            )
                        )
                        if lrule not in classes:
                            classes.append(lrule)
        return classes

    def _get_data(self, train):
        """Loads in the conllu data and prepares it as a list dataset so that it can be indexed for __getitem__"""
        data = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                sentence = tk_list.metadata["text"]
                uposes = []
                xposes = []
                lemma_rules = []
                ufeats = []
                for i, tok in enumerate(tk_list):
                    if isinstance(tok["id"], tuple):
                        if train:
                            # This is a multi-word token, id looks like (1, '-', 3)
                            # creating a look-up table to be used when loading in sentences
                            start, sep, end = tok["id"] # these are not perfect indices because of the multiword token stuff
                            if not sep == '.':
                                # when sep is ., this is an ellided token, not a multiword token
                                x = (end - start) + i
                                # below, i + 1 is because we want the next token, not the current
                                words = [t["form"] for t in tk_list[i + 1: x + 2]]
                                if tok["form"] in self.multiword_table:
                                    self.multiword_table[tok["form"]]["frequency"] += 1
                                else:
                                    self.multiword_table[tok["form"]] = {"frequency": 1, "words": words}
                        continue  # we don't want to add it to the dataset!
                    l_rule = str(
                        self.e_script(
                            tok["form"].lower(), tok["lemma"].lower()
                        )
                    )
                    # Label encoders can't handle None!
                    if tok["feats"] is None:
                        tok["feats"] = "_"

                    # Here we have to check if the thing is unknown
                    upos_ = tok["upos"] if tok["upos"] in self.upos_encoder.classes_ else self.UNK_TAG
                    xpos_ = tok["xpos"] if tok["xpos"] in self.xpos_encoder.classes_ else self.UNK_TAG
                    ufeat_ = tok["feats"] if tok["feats"] in self.ufeats_encoder.classes_ else self.UNK_TAG
                    if l_rule not in self.lemma_encoder.classes_:
                        l_rule = self.UNK_TAG
                    uposes.append(upos_)
                    xposes.append(xpos_)
                    lemma_rules.append(l_rule)
                    ufeats.append(ufeat_)
                    self.all_words.append(tok["form"].lower())
                uposes = self.upos_encoder.transform(uposes)
                xposes = self.xpos_encoder.transform(xposes)
                lemma_rules = self.lemma_encoder.transform(lemma_rules)
                ufeats = self.ufeats_encoder.transform(ufeats)
                # heads do not need to be encoded, they are already numerical. Encoding them here causes problems later
                data.append((sentence, uposes, xposes, lemma_rules, ufeats))
            if train:
                self.check_multiword_table()
                with open(f'{self.path_name}/multiword_dict.json', 'w') as mw_tb:
                    json.dump(self.multiword_table, mw_tb, ensure_ascii=False)
        return data

    def check_multiword_table(self):
        """Make sure all items are valid. Some entries could be typos or annotator mistakes,
        if it is more frequent as a single word, it is assumed to be a typo/annotator mistake."""
        single_cnt = Counter(self.all_words)
        keep = {}
        for w, record in self.multiword_table.items():
            w = w.lower()
            if w in single_cnt and single_cnt[w] > record["frequency"]:
                continue
            keep[w] = record
        self.multiword_table = keep


    def get_special_words(self):
        """Special words are ones that are not handled well by tokenizers, i.e. 2000-2004 or K., which are considered
        single tokens in some conllu files."""
        special_words = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tk in tk_list:
                    if isinstance(tk["id"], int) and re.search(r"\w+[.'-\\/%]|[.'-\\/%]\w+", tk["form"]):
                        special_words.append(tk["form"])
        return special_words

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


class TextIterDataset(IterableDataset):
    """Iterable dataset, used for inference when labels are unknown

    This class is used when the data for inference is large and we do not want to load it entirely into memory.
    """
    def __init__(self, text_file: str):
        """Initializes the instance based on user input.

        Args:
            text_file: The path to the textfile to incrementally read from.
        """
        super().__init__()
        if text_file:
            self.tf = open(text_file)

    def __iter__(self):
        return self.tf

    def __del__(self):
        if hasattr(self, "tf"):
            self.tf.close()  # want to make sure we close the file during garbage collection
