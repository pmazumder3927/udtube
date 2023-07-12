import argparse
from typing import Iterable

import pytorch_lightning as pl
import torch
import transformers
from torch import nn, tensor
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall

from batch import InferenceBatch, TrainBatch
from conllu_datasets import UPOS_CLASSES, TextIterDataset, ConlluMapDataset
from data_utils import inference_preprocessing, train_preprocessing
from functools import partial


def set_up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A trainable model for producing morphological analysis for UD files"
    )
    parser.add_argument(
        "dataset_path",
        help="The file path of the input conllu file to be used for training "
        "or inference",
    )
    parser.add_argument(
        "model_name",
        help="The name of the model. If training, this makes a new file with "
        "this name. At inference",
    )
    parser.add_argument(
        "procedure",
        choices=["train", "evaluate", "inference"],
        help="What will be done on this run of the model. "
        'Choices are ["train", "evaluate", "inference"]',
    )
    parser.add_argument(
        "--out_file_path",
        default="udtube_out.conllu",
        help="The output file path if you are inferencing",
    )
    parser.add_argument(
        "--inference_input_type",
        default="txt",
        choices=["txt", "conllu"],
        help="The input file type for inferencing. Choices are either txt (a txt file with a sentence on each line) "
             "or conllu."
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-multilingual-cased",
        help="The base BERT model to be fine-tuned.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="The number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="The batch size for the bert model",
    )
    parser.add_argument(
        "--manual_seed",
        default=42,
        type=int,
        help="The random seed.",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="The learning rate for the optimizer",
    )
    parser.add_argument(
        "--reverse",
        action='store_true',
        default=True,
        help="Reverse edit script calculation. Recommended for suffixal languages. True by default.",
    )
    parser.add_argument(
        "--no_reverse",
        action='store_false',
        dest="reverse",
        help="Left to right edit script calculation. Recommended for non-suffixal languages.",
    )
    parser.add_argument(
        "--gold_file",
        help="To be used with procedure=evaluate, this is the gold file.",
    )
    parser.add_argument(
        "--pred_file",
        help="To be used with procedure=evaluate, this is the pred file.",
    )
    parser.add_argument(
        "--pooling_layers",
        default=4,
        type=int,
        help="The amount of layers we pull embeddings from. Default is 4."
    )
    # TODO add other hyper params
    return parser


class UDTube(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pos_out_label_size: int = 2,
        lemma_out_label_size: int = 2,
        ufeats_out_label_size: int = 2,
        learning_rate: float = 0.001,
        pooling_layers: int = 4
    ):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.learning_rate = learning_rate
        self.pooling_layers = pooling_layers
        self.pos_pad = tensor(
            pos_out_label_size - 1
        )  # last item in the list is a pad from the label encoder
        self.lemma_pad = tensor(lemma_out_label_size)
        self.ufeats_pad = tensor(ufeats_out_label_size)
        self.pos_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, pos_out_label_size
            ),
            nn.Tanh()
        )
        self.lemma_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, lemma_out_label_size + 1
            ),  # + 1 for padding labels (for now)
            nn.Tanh(),
        )
        self.ufeats_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, ufeats_out_label_size + 1
            ),  # + 1 for padding labels (for now)
            nn.Tanh(),
        )

        # Setting up all the metrics objects for each task
        self.pos_loss = nn.CrossEntropyLoss(ignore_index=self.pos_pad.item())
        self.pos_precision = Precision(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=self.pos_pad.item(),
        )
        self.pos_recall = Recall(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=self.pos_pad.item(),
        )
        self.pos_f1 = F1Score(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=self.pos_pad.item(),
        )
        self.pos_accuracy = Accuracy(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=self.pos_pad.item(),
        )

        self.lemma_loss = nn.CrossEntropyLoss(
            ignore_index=self.lemma_pad.item()
        )
        self.lemma_precision = Precision(
            task="multiclass",
            num_classes=lemma_out_label_size + 1,
            ignore_index=self.lemma_pad.item(),
        )
        self.lemma_recall = Recall(
            task="multiclass",
            num_classes=lemma_out_label_size + 1,
            ignore_index=self.lemma_pad.item(),
        )
        self.lemma_f1 = F1Score(
            task="multiclass",
            num_classes=lemma_out_label_size + 1,
            ignore_index=self.lemma_pad.item(),
        )
        self.lemma_accuracy = Accuracy(
            task="multiclass",
            num_classes=lemma_out_label_size + 1,
            ignore_index=self.lemma_pad.item(),
        )

        self.ufeats_loss = nn.CrossEntropyLoss(
            ignore_index=self.ufeats_pad.item()
        )
        self.ufeats_precision = Precision(
            task="multiclass",
            num_classes=ufeats_out_label_size + 1,
            ignore_index=self.ufeats_pad.item(),
        )
        self.ufeats_recall = Recall(
            task="multiclass",
            num_classes=ufeats_out_label_size + 1,
            ignore_index=self.ufeats_pad.item(),
        )
        self.ufeats_f1 = F1Score(
            task="multiclass",
            num_classes=ufeats_out_label_size + 1,
            ignore_index=self.ufeats_pad.item(),
        )
        self.ufeats_accuracy = Accuracy(
            task="multiclass",
            num_classes=ufeats_out_label_size + 1,
            ignore_index=self.ufeats_pad.item(),
        )

    def pad_seq(
        self,
        sequence: Iterable,
        pad: Iterable,
        max_len: int,
        return_long: bool = False,
    ):
        padded_seq = []
        for s in sequence:
            if len(s) != max_len:
                r_padding = torch.stack([pad] * (max_len - len(s)))
                padded_seq.append(torch.cat((s, r_padding)))
            else:
                padded_seq.append(s)
        if return_long:
            return torch.stack(padded_seq).long()
        return torch.stack(padded_seq)

    def pool_embeddings(
        self, x_embs: torch.tensor, tokenized: transformers.BatchEncoding
    ):
        new_embs = []
        new_masks = []
        for encoding, x_emb_i in zip(tokenized.encodings, x_embs):
            embs_i = []
            mask_i = []
            last_word_idx = slice(0, 0)
            for word_id, x_emb_j in zip(encoding.word_ids, x_emb_i):
                if word_id is None:
                    embs_i.append(x_emb_j)
                    # TODO maybe make dummy tensor a member of class
                    dummy_tensor = x_emb_j
                    mask_i.append(0)
                    continue
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    word_emb_pooled = torch.mean(
                        x_emb_i[word_idxs], keepdim=True, dim=0
                    ).squeeze()
                    embs_i.append(word_emb_pooled)
                    mask_i.append(1)
            new_embs.append(torch.stack(embs_i))
            new_masks.append(tensor(mask_i))
        longest_seq = max(len(m) for m in new_masks)
        new_embs = self.pad_seq(new_embs, dummy_tensor, longest_seq)
        new_masks = self.pad_seq(new_masks, tensor(0), longest_seq)
        return new_embs, new_masks, longest_seq

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        batch_size: int,
        task_name: str,
        subset: str = "train",
    ):
        precision = getattr(self, f"{task_name}_precision")
        recall = getattr(self, f"{task_name}_recall")
        f1 = getattr(self, f"{task_name}_f1")
        accuracy = getattr(self, f"{task_name}_accuracy")

        avg_precision = precision(y_pred, y_true)
        avg_recall = recall(y_pred, y_true)
        avg_f1s = f1(y_pred, y_true)
        avg_accs = accuracy(y_pred, y_true)

        self.log(
            f"{subset}:{task_name}_precision",
            avg_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        self.log(
            f"{subset}:{task_name}_recall",
            avg_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        self.log(
            f"{subset}:{task_name}_f1",
            avg_f1s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        self.log(
            f"{subset}:{task_name}_acc",
            avg_accs,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

    def forward(self, batch):
        x_encoded = self.bert(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_n_layer_embs = torch.stack(x_encoded.hidden_states[-self.pooling_layers:])
        x_embs = torch.mean(last_n_layer_embs, keepdim=True, dim=0).squeeze()
        x_word_embs, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens
        )

        y_pos_logits = self.pos_head(x_word_embs)
        y_lemma_logits = self.lemma_head(x_word_embs)
        y_ufeats_logits = self.ufeats_head(x_word_embs)

        return y_pos_logits, y_lemma_logits, y_ufeats_logits

    def training_step(self, batch, batch_idx: int, subset: str = "train"):
        x_encoded = self.bert(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_4_layer_embs = torch.stack(x_encoded.hidden_states[-4:])
        x_embs = torch.mean(last_4_layer_embs, keepdim=True, dim=0).squeeze()
        x_word_embs, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens
        )

        # need to do some preprocessing on Y
        y_pos_tensor = self.pad_seq(
            batch.pos, self.pos_pad, longest_seq, return_long=True
        )  # TODO passing self. is weird
        y_lemma_tensor = self.pad_seq(
            batch.lemmas, self.lemma_pad, longest_seq, return_long=True
        )
        y_ufeats_tensor = self.pad_seq(
            batch.ufeats, self.ufeats_pad, longest_seq, return_long=True
        )

        # getting logits from each head, and then permuting them for metrics calculation
        # Each head returns batch X sequence_len X classes
        # but CE and metrics want minibatch X Classes X sequence_len, (minibatch, C, d0...dk) & (N, C, ..) in the docs.
        y_pos_logits = self.pos_head(x_word_embs)
        y_pos_logits = y_pos_logits.permute(0, 2, 1)

        y_lemma_logits = self.lemma_head(x_word_embs)
        y_lemma_logits = y_lemma_logits.permute(0, 2, 1)

        y_ufeats_logits = self.ufeats_head(x_word_embs)
        y_ufeats_logits = y_ufeats_logits.permute(0, 2, 1)

        # getting loss and logging for each head
        batch_size = len(batch)

        pos_loss = self.pos_loss(y_pos_logits, y_pos_tensor)
        self.log_metrics(y_pos_logits, y_pos_tensor, batch_size, "pos", subset=subset)

        lemma_loss = self.lemma_loss(y_lemma_logits, y_lemma_tensor)
        self.log_metrics(
            y_lemma_logits, y_lemma_tensor, batch_size, "lemma", subset=subset
        )

        ufeats_loss = self.ufeats_loss(y_ufeats_logits, y_ufeats_tensor)
        self.log_metrics(
            y_ufeats_logits, y_ufeats_tensor, batch_size, "ufeats", subset=subset
        )

        # combining the loss of the heads
        loss = torch.mean(torch.stack([pos_loss, lemma_loss, ufeats_loss]))
        self.log(
            "Loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        return self.training_step(batch, batch_idx, subset="val")


if __name__ == "__main__":
    parser = set_up_parser()
    # TODO remove those args, they are for developing purposes
    args = parser.parse_args("el_gdt-ud-dev.conllu el_ud_tube train --reverse".split())

    # initializing a tokenizer for preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model)

    if args.procedure == "train":
        # Loading in the data
        data = ConlluMapDataset(args.dataset_path, reverse_edits=True)

        # making a validation set
        seed_for_reproducibility = torch.Generator().manual_seed(
            args.manual_seed
        )
        train_data, val_data = random_split(
            data, [0.8, 0.2], seed_for_reproducibility
        )

        # Loading the data into a dataloader with a preprocessing function to make X a tensor
        train_dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            collate_fn=partial(train_preprocessing, tokenizer=tokenizer),
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            collate_fn=partial(train_preprocessing, tokenizer=tokenizer),
        )

        # initializing the model
        pos_out_label_size = len(UPOS_CLASSES)
        ufeats_out_label_size = len(data.feats_classes)
        lemma_out_label_size = len(data.lemma_classes)
        model = UDTube(
            args.bert_model,
            pos_out_label_size=pos_out_label_size,
            lemma_out_label_size=lemma_out_label_size,
            ufeats_out_label_size=ufeats_out_label_size,
            learning_rate=args.learning_rate,
            pooling_layers=args.pooling_layers
        )

        # Training the model
        trainer = pl.Trainer(max_epochs=args.epochs)
        trainer.fit(model, train_dataloader, val_dataloader)

    if args.procedure == "inference":
        if args.inference_input_type == "txt":
            tdata = TextIterDataset(args.dataset_path)
        else:
            raise NotImplementedError(f"The inference input type: {args.inference_input_type} is not implemented just yet")
        test_dataloader = DataLoader(
            tdata,
            batch_size=args.batch_size,
            collate_fn=partial(inference_preprocessing, tokenizer=tokenizer),
        )
        trainer.predict(model, test_dataloader)

    if args.produce == "evaluate":
        gold_file_path = args.gold_file
        pred_file_path = args.pred_file
        raise NotImplementedError("Evaluation is not yet implemented!")
