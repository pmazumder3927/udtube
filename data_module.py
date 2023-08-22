from functools import partial
import os
import torch
import glob

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from batch import ConlluBatch, TextBatch
from conllu_datasets import ConlluMapDataset, TextIterDataset


def get_most_recent_file(fp, t='dir'):
    if t == 'dir':
        s = '*/'
    else:
        s = '*.ckpt'
    # citation: https://stackoverflow.com/questions/2014554/find-the-newest-folder-in-a-directory-in-python
    return max(glob.glob(os.path.join(fp, s)), key=os.path.getmtime)


class ConlluDataModule(pl.LightningDataModule):
    """Reusable DataModule that manages stages of UDTube

    This class is initialized by the LightningCLI interface. It manages all the data loading steps for UDTube. Train/Val
    are loaded into memory, but prediction data is loaded in slowly to avoid memory errors when loading in large files.
    """
    def __init__(
        self,
        path_name: str = "UDTube",
        model_name: str = None,
        train_dataset: str = None,
        val_dataset: str = None,
        predict_dataset: str = None,
        test_dataset: str = None,
        batch_size: int = 32,
        reverse_edits: bool = False,
    ):
        """Initializes the instance based on user input. Some attributes will not be set if train dataset is None.

        Args:
            path_name: The name of the folder to save model assets in
            model_name: The name of the model; used to tokenize and encode.
            train_dataset: The path to the training conllu file
            val_dataset: The path to the validation conllu file
            predict_dataset: The path to the prediction dataset, can either be text or conllu
            batch_size: The batch_size
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
        """
        super().__init__()
        self.reverse_edits = reverse_edits
        self.train_dataset = ConlluMapDataset(train_dataset, reverse_edits=self.reverse_edits, path_name=path_name)
        self.val_dataset = ConlluMapDataset(val_dataset, reverse_edits=self.reverse_edits, path_name=path_name)
        self.predict_dataset = TextIterDataset(predict_dataset)
        self.test_dataset = ConlluMapDataset(test_dataset)
        self.batch_size = batch_size
        self.tokenizer_name = model_name
        if self.train_dataset:
            # this is a bit hacky, but not sure how to do this with setup & CLI
            # + 1 is the padding & unk tok
            self.pos_classes_cnt = len(self.train_dataset.UPOS_CLASSES) + 2
            self.lemma_classes_cnt = len(self.train_dataset.lemma_classes) + 2
            self.feats_classes_cnt = len(self.train_dataset.feats_classes) + 2
        else:
            self._set_values_from_path_name(path_name)

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _set_values_from_path_name(self, path_name):
        if path_name and os.path.exists(f"{path_name}/lightning_logs"):
            # for now, the version we get is the newest one from the logs
            # citation: https://stackoverflow.com/questions/2014554/find-the-newest-folder-in-a-directory-in-python
            most_recent_ver = get_most_recent_file(f"{path_name}/lightning_logs", t='dir')
            if not os.path.exists(f"{most_recent_ver}checkpoints"):
                raise Exception("The most recent model version in the logs does not have a checkpoint file.")
            most_recent_ckpt = get_most_recent_file(f"{most_recent_ver}checkpoints", t='file')
            # This is sloppy, but it keeps things consistent for prediction
            hps = torch.load(most_recent_ckpt)["hyper_parameters"]
            self.pos_classes_cnt = hps.get("pos_out_label_size", 0)
            self.lemma_classes_cnt = hps.get("lemma_out_label_size", 0)
            self.feats_classes_cnt = hps.get("feats_out_label_size", 0)

    @staticmethod
    def conllu_preprocessing(batch, tokenizer: AutoTokenizer):
        """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
        sentences, *args = zip(*batch)
        # I want to load in tokens by batch to avoid using the inefficient max_length padding
        tokenized_x = tokenizer(
            sentences, padding="longest", return_tensors="pt"
        ).to("cuda")
        return ConlluBatch(tokenized_x, sentences, *args)

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

    @staticmethod
    def txt_file_preprocessing(sentences, tokenizer: AutoTokenizer):
        """This is the case where the input is NOT a conllu file"""
        tokenized_x = tokenizer(
            sentences, padding="longest", return_tensors="pt"
        ).to("cuda")
        return TextBatch(tokenized_x, sentences)

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
