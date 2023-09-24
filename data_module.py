import json
import re
from functools import partial
import os
import torch
import spacy_udpipe

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5TokenizerFast
from tokenizers import Encoding

from batch import ConlluBatch, TextBatch
from conllu_datasets import ConlluMapDataset, TextIterDataset


class CustomEncoding():
    def __init__(self, word_ids, tokens):
        self.word_ids = word_ids
        self.tokens = tokens

    def word_to_tokens(self, word_id):
        # TODO: less naive approach
        start = self.word_ids.index(word_id)
        stop = start + 1
        for i, idx in enumerate(self.word_ids[start + 1:]):
            if idx == word_id:
                stop += 1
            else:
                break
        return start, stop



class ConlluDataModule(pl.LightningDataModule):
    """Reusable DataModule that manages stages of UDTube

    This class is initialized by the LightningCLI interface. It manages all the data loading steps for UDTube. Train/Val
    are loaded into memory, but prediction data is loaded in slowly to avoid memory errors when loading in large files.
    """
    def __init__(
        self,
        path_name: str = "UDTube",
        model_name: str = None,
        language: str = None,
        train_dataset: str = None,
        val_dataset: str = None,
        predict_dataset: str = None,
        test_dataset: str = None,
        batch_size: int = 32,
        convert_to_um: bool = True,
        reverse_edits: bool = False,
        checkpoint: str = None
    ):
        """Initializes the instance based on user input. Some attributes will not be set if train dataset is None.

        Args:
            path_name: The name of the folder to save model assets in
            model_name: The name of the model; used to tokenize and encode.
            train_dataset: The path to the training conllu file
            val_dataset: The path to the validation conllu file
            predict_dataset: The path to the prediction dataset, can either be text or conllu
            batch_size: The batch_size
            convert_to_um: enable Universal Dependency (UD) file conversion to Universal Morphology (UM) format
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
            checkpoint: The path to the model checkpoint file
        """
        super().__init__()
        spacy_udpipe.download(language)
        self.pretokenizer = spacy_udpipe.load(language)
        self.reverse_edits = reverse_edits
        self.train_dataset = ConlluMapDataset(train_dataset, reverse_edits=self.reverse_edits, path_name=path_name,
                                              convert_to_um=convert_to_um, train=True)
        self.val_dataset = ConlluMapDataset(val_dataset, reverse_edits=self.reverse_edits, path_name=path_name,
                                            convert_to_um=convert_to_um, train=False)
        self.predict_dataset = TextIterDataset(predict_dataset)
        self.test_dataset = ConlluMapDataset(test_dataset, convert_to_um=convert_to_um)
        with open(f"{path_name}/multiword_dict.json", "r") as mw_tb:
            self.multiword_table = json.load(mw_tb)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.checkpoint = checkpoint
        if self.train_dataset:
            # this is a bit hacky, but not sure how to do this with setup & CLI
            # + 2 is the padding & unk tok
            self.pos_classes_cnt = len(self.train_dataset.UPOS_CLASSES) + 2
            self.xpos_classes_cnt = len(self.train_dataset.xpos_classes) + 2
            self.lemma_classes_cnt = len(self.train_dataset.lemma_classes) + 2
            self.feats_classes_cnt = len(self.train_dataset.feats_classes) + 2
            self.deprel_classes_cnt = len(self.train_dataset.deprel_classes) + 2
        else:
            self._set_values_from_path_name()

    def _set_values_from_path_name(self):
        assert self.checkpoint, "model checkpoint must not be none"
        hps = torch.load(self.checkpoint)["hyper_parameters"]
        self.pos_classes_cnt = hps.get("pos_out_label_size", 0)
        self.lemma_classes_cnt = hps.get("lemma_out_label_size", 0)
        self.feats_classes_cnt = hps.get("feats_out_label_size", 0)
        self.deprel_classes_cnt = hps.get("deprel_out_label_size", 0)

    def _adjust_sentence(self, sentences, lang_with_space=True):
        delimiter = " " if lang_with_space else ""  # TODO does this actually make sense? Will find out when I try zh
        new_sents = []
        replacements = []  # will need to keep track of these to write out multiword tokens
        # preprocessing sentences. The table is usually small so this is not too expensive
        for s in sentences:
            ns = s
            rep_i = []
            for k, v in self.multiword_table.items():
                if re.search(fr"\b{k}\b", s):
                    ns = re.sub(fr"\b{k}\b", delimiter.join(v), ns)
                    rep_i.append((k, delimiter.join(v)))
            new_sents.append(ns)
            replacements.append(rep_i)
        return new_sents, replacements

    def _pretokenize(self, sentences):
        batch_toks = []
        for s in sentences:
            toks = [t.text for t in self.pretokenizer(s)]
            batch_toks.append(toks)
        return batch_toks

    def conllu_preprocessing(self, batch, tokenizer: AutoTokenizer, lang_with_space=True):
        """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
        sentences, *args = zip(*batch)
        sentences, replacements = self._adjust_sentence(sentences, lang_with_space=lang_with_space)
        pretoks = self._pretokenize(sentences)
        # I want to load in tokens by batch to avoid using the inefficient max_length padding
        tokenized_x = tokenizer(
            pretoks, padding="longest", return_tensors="pt", is_split_into_words=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if not tokenized_x.encodings:
            # for Byt5, which doesn't seem to have a fast tokenizer
            encodings_ = []
            for sent in pretoks:
                idxs = []
                toks = []
                for i, word in enumerate(sent):
                    # number of toks
                    toks_ = list(word.encode("utf-8"))
                    num_toks = len(toks_)
                    idxs.extend([i] * num_toks)
                    toks.extend(toks_)
                # this is not the prettiest, but Encoding does not have an __init__(), so this is set manually
                encodings_.append(CustomEncoding(word_ids=idxs, tokens=toks))
            tokenized_x.custom_encodings = encodings_

        return ConlluBatch(tokenized_x, sentences, replacements, *args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.conllu_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.conllu_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.conllu_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def txt_file_preprocessing(self, sentences, tokenizer: AutoTokenizer, lang_with_space=True):
        """This is the case where the input is NOT a conllu file"""
        sentences, replacements = self._adjust_sentence(sentences, lang_with_space=lang_with_space)
        pretoks = self._pretokenize(sentences)
        tokenized_x = tokenizer(
            pretoks, padding="longest", return_tensors="pt", is_split_into_words=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        return TextBatch(tokenized_x, sentences, replacements)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.txt_file_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def teardown(self, stage: str) -> None:
        if stage == "predict":
            del self.predict_dataset
