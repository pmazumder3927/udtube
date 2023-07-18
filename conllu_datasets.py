import conllu
import torch
from sklearn.preprocessing import LabelEncoder
from torch import tensor
from torch.utils.data import Dataset, IterableDataset

import edit_scripts

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
    "_",
    PAD_TAG,
]
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
    """Conllu format Dataset. It loads the entire dataset into memory and therefore is only suitable for smaller
    datasets"""

    def __init__(self, conllu_file: str, reverse_edits: bool = False):
        super().__init__()
        self.conllu_file = conllu_file
        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        # setting up label encoders
        if conllu_file:
            self.upos_encoder = LabelEncoder()
            self.ufeats_encoder = LabelEncoder()
            self.lemma_encoder = LabelEncoder()
            self.feats_classes = self._get_all_classes("feats")
            self.lemma_classes = self._get_all_classes("lemma")
            self._fit_label_encoders()
            self.data_set = self._get_data()

    def _fit_label_encoders(self):
        self.upos_encoder.fit(UPOS_CLASSES)
        self.ufeats_encoder.fit(self.feats_classes)
        self.lemma_encoder.fit(self.lemma_classes)

    def _get_all_classes(self, lname: str):
        """helper function to get all the classes observed in the training set"""
        classes = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tok in tk_list:
                    if lname != "lemma" and tok[lname] not in classes:
                        classes.append(tok[lname])
                    elif lname == "lemma":
                        lrule = str(
                            self.e_script(
                                tok["form"].lower(), tok[lname].lower()
                            )
                        )
                        if lrule not in classes:
                            classes.append(lrule)
        return classes

    def _get_data(self):
        """Loads in the conllu data and prepares it as a list dataset so that it can be indexed for __getitem__"""
        data = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                sentence = tk_list.metadata["text"]
                tokens = []
                uposes = []
                lemma_rules = []
                ufeats = []
                for tok in tk_list:
                    l_rule = str(
                        self.e_script(
                            tok["form"].lower(), tok["lemma"].lower()
                        )
                    )
                    tokens.append(tok["form"])
                    uposes.append(tok["upos"])
                    lemma_rules.append(l_rule)
                    ufeats.append(tok["feats"])
                uposes = tensor(self.upos_encoder.transform(uposes))
                lemma_rules = tensor(self.lemma_encoder.transform(lemma_rules))
                ufeats = tensor(self.ufeats_encoder.transform(ufeats))
                data.append((sentence, tokens, uposes, lemma_rules, ufeats))
        return data

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


class TextIterDataset(IterableDataset):
    """Iterable dataset, used for inference when labels are unknown"""

    def __init__(self, text_file: str):
        super().__init__()
        if text_file:
            self.tf = open(text_file)

    def __iter__(self):
        return self.tf

    def __del__(self):
        if hasattr(self, "tf"):
            self.tf.close()
