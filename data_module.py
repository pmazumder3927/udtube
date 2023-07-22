from functools import partial

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from batch import PredictBatch, TrainBatch
from conllu_datasets import ConlluMapDataset, TextIterDataset


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
        self.batch_size = batch_size
        self.tokenizer_name = model_name
        if self.train_dataset:
            # this is a bit hacky, but not sure how to do this with setup & CLI
            # + 1 is the padding
            self.pos_classes_cnt = len(self.train_dataset.UPOS_CLASSES) + 1
            self.lemma_classes_cnt = len(self.train_dataset.lemma_classes) + 1
            self.feats_classes_cnt = len(self.train_dataset.feats_classes) + 1

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    @staticmethod
    def train_preprocessing(batch, tokenizer: AutoTokenizer):
        """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
        sentences, *args = zip(*batch)
        # I want to load in tokens by batch to avoid using the inefficient max_length padding
        tokenized_x = tokenizer(
            sentences, padding="longest", return_tensors="pt"
        )
        return TrainBatch(tokenized_x, sentences, *args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.train_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.train_preprocessing, tokenizer=self.tokenizer
            ),
        )

    @staticmethod
    def txt_inference_preprocessing(sentences, tokenizer: AutoTokenizer):
        """This is the case where the input is NOT a conllu file"""
        tokenized_x = tokenizer(
            sentences, padding="longest", return_tensors="pt"
        )
        return PredictBatch(tokenized_x, sentences)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.txt_inference_preprocessing, tokenizer=self.tokenizer
            ),
        )

    def teardown(self, stage: str) -> None:
        if stage == "predict":
            del self.predict_dataset
