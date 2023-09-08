import os
import re

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
        "_",
    ]

    def __init__(self, conllu_file: str, reverse_edits: bool = False, path_name: str = "UDTube",
                 convert_to_um: bool = True, fit_encoders: bool = False):
        """Initializes the instance based on user input.

        Args:
            conllu_file: The path to the conllu file to make the dataset from
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
            path_name: The dir where everything will get saved
            convert_to_um: Enable Universal Dependency (UD) file conversion to Universal Morphology (UM) format
            fit_encoders: Fit the label encoders. Should only be true for the train dataset
        """
        super().__init__()
        self.path_name = path_name
        self.conllu_file = self._convert_to_um(conllu_file, convert_to_um)
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
            self.deprel_encoder = LabelEncoder()
            self.feats_classes = self._get_all_classes("feats")
            self.lemma_classes = self._get_all_classes("lemma")
            self.deprel_classes = self._get_all_classes("deprel")
            if fit_encoders:
                self._fit_label_encoders()
            else:
                self._load_label_encoders()
            self.data_set = self._get_data()
        else:
            # Instatiation of empty class, happens in prediction
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
        # this ensures that the PAD ends up last and the UNK ends up at 0
        self.upos_encoder.fit([self.UNK_TAG] + self.UPOS_CLASSES + [self.PAD_TAG])
        self.ufeats_encoder.fit([self.UNK_TAG] + self.feats_classes + [self.PAD_TAG])
        self.lemma_encoder.fit([self.UNK_TAG] + self.lemma_classes + [self.PAD_TAG])
        self.deprel_encoder.fit([self.UNK_TAG] + self.deprel_classes + [self.PAD_TAG])
        # saving all the encoders
        if not os.path.exists(self.path_name):
            os.mkdir(self.path_name)
        print(f"Now saving label encoders for {self.conllu_file}")
        joblib.dump(self.upos_encoder, f'{self.path_name}/upos_encoder.joblib')
        joblib.dump(self.ufeats_encoder, f'{self.path_name}/ufeats_encoder.joblib')
        joblib.dump(self.lemma_encoder, f'{self.path_name}/lemma_encoder.joblib')
        joblib.dump(self.deprel_encoder, f'{self.path_name}/deprel_encoder.joblib')

    def _load_label_encoders(self):
        self.lemma_encoder = joblib.load(f"{self.path_name}/lemma_encoder.joblib")
        self.ufeats_encoder = joblib.load(f"{self.path_name}/ufeats_encoder.joblib")
        self.upos_encoder = joblib.load(f"{self.path_name}/upos_encoder.joblib")
        self.deprel_encoder = joblib.load(f"{self.path_name}/deprel_encoder.joblib")

    def _get_all_classes(self, lname: str):
        """helper function to get all the classes observed in the training set"""
        classes = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tok in tk_list:
                    if tok[lname] and lname != "lemma" and tok[lname] not in classes:
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
                uposes = []
                lemma_rules = []
                ufeats = []
                heads = []
                deprels = []
                for tok in tk_list:
                    if isinstance(tok["id"], tuple):
                        # This is a multiword token. For now, I am just skipping adding them to the data
                        continue
                    l_rule = str(
                        self.e_script(
                            tok["form"].lower(), tok["lemma"].lower()
                        )
                    )
                    # Here we have to check if the thing is unknown
                    upos_ = tok["upos"] if tok["upos"] in self.upos_encoder.classes_ else self.UNK_TAG
                    ufeat_ = tok["feats"] if tok["feats"] in self.ufeats_encoder.classes_ else self.UNK_TAG
                    deprel_ = tok["deprel"] if tok['deprel'] in self.deprel_encoder.classes_ else self.UNK_TAG
                    # heads is a little different
                    head_ = tok["head"] if isinstance(tok["head"], int) else len(tk_list) # the sequence length is the unknown tag
                    if l_rule not in self.lemma_encoder.classes_:
                        l_rule = self.UNK_TAG
                    uposes.append(upos_)
                    lemma_rules.append(l_rule)
                    ufeats.append(ufeat_)
                    heads.append(head_)
                    deprels.append(deprel_)
                uposes = self.upos_encoder.transform(uposes)
                lemma_rules = self.lemma_encoder.transform(lemma_rules)
                ufeats = self.ufeats_encoder.transform(ufeats)
                deprels = self.deprel_encoder.transform(deprels)
                # heads do not need to be encoded, they are already numerical. Encoding them here causes problems later
                heads = np.array(heads).T
                data.append((sentence, uposes, lemma_rules, ufeats, heads, deprels))
        return data

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
