from typing import Iterable

import lightning.pytorch as pl
import torch
import transformers
import joblib
from mst import mst

import edit_scripts
from lightning.pytorch.cli import LightningCLI
from torch import nn, tensor
from torchmetrics.functional.classification import multiclass_accuracy

from batch import ConlluBatch, TextBatch
from callbacks import CustomWriter
from data_module import ConlluDataModule
from biaffine_parser import BiaffineParser
from typing import Union


class UDTubeCLI(LightningCLI):
    """A customized version of the Lightning CLI

    This class manages all processes behind the scenes, to find out what it can do, run python udtube.py --help
    """

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--output_file")
        parser.link_arguments("model.path_name", "data.path_name")
        parser.link_arguments("model.path_name", "trainer.default_root_dir")
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.reverse_edits", "model.reverse_edits")
        parser.link_arguments(
            "data.pos_classes_cnt",
            "model.pos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.xpos_classes_cnt",
            "model.xpos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.lemma_classes_cnt",
            "model.lemma_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_classes_cnt",
            "model.feats_out_label_size",
            apply_on="instantiate",
        )

    def before_instantiate_classes(self) -> None:
        if self.subcommand == "predict":
            self.trainer_defaults["callbacks"] = [CustomWriter(self.config.predict.output_file)]
        elif self.subcommand == "test":
            self.trainer_defaults["callbacks"] = [CustomWriter(self.config.test.output_file)]


class UDTube(pl.LightningModule):
    """The main model file

    UDTube is a BERT based model that handles 3 tasks at once, pos tagging, lemmatization, and feature classification.
    """

    def __init__(
            self,
            path_name: str = "UDTube",
            model_name: str = "bert-base-multilingual-cased",
            pos_out_label_size: int = 2,
            xpos_out_label_size: int = 2,
            lemma_out_label_size: int = 2,
            feats_out_label_size: int = 2,
            udtube_dropout: float = 0.3,
            encoder_dropout: float = 0.5,
            udtube_learning_rate: float = 5e-3,
            encoder_model_learning_rate: float = 2e-5,
            pooling_layers: int = 4,
            reverse_edits: bool = False,
            checkpoint: str = None,
            pos_toggle: bool = True,
            xpos_toggle: bool = True,
            lemma_toggle: bool = True,
            feats_toggle: bool = True
    ):
        """Initializes the instance based on user input.

        Args:
            path_name: The name of the folder to save/load model assets. This includes the lightning logs and the encoders.
            model_name: The name of the model; used to tokenize and encode.
            pos_out_label_size: The amount of POS labels. This is usually passed by the dataset.
            xpos_out_label_size: The amount of language specific POS labels. This is usually passed by the dataset.
            lemma_out_label_size: The amount of lemma rule labels in the dataset. This is usually passed by the dataset.
            feats_out_label_size: The amount of feature labels in the dataset. This is usually passed by the dataset.
            udtube_learning_rate: The learning rate of the full model
            encoder_model_learning_rate: The learning rate of only the encoder
            pooling_layers: The amount of layers used for embedding calculation
            checkpoint: The model checkpoint file
            pos_toggle: Whether the model will do POS tagging or not
            xpos_toggle: Whether not the model will do XPOS tagging or not
            lemma_toggle: Whether the model will do lemmatization or not
            feats_toggle: Whether the model will do Ufeats tagging or not
        """
        super().__init__()
        self._validate_input(
            pos_out_label_size, lemma_out_label_size, feats_out_label_size
        )
        self.path_name = path_name
        self.udtube_learning_rate = udtube_learning_rate
        self.encoder_model_learning_rate = encoder_model_learning_rate
        self.encoder_dropout = encoder_dropout
        self.pooling_layers = pooling_layers

        self.encoder_model = self._load_model(model_name)
        if pos_toggle:
            self.pos_head = nn.Sequential(
                nn.Linear(self.encoder_model.config.hidden_size, pos_out_label_size),
                nn.LeakyReLU()
            )
        if xpos_toggle:
            self.xpos_head = nn.Sequential(
                nn.Linear(self.encoder_model.config.hidden_size, xpos_out_label_size),
                nn.LeakyReLU()
            )
        if lemma_toggle:
            self.lemma_head = nn.Sequential(
                nn.Linear(self.encoder_model.config.hidden_size, lemma_out_label_size),
                nn.LeakyReLU()
            )
        if feats_toggle:
            self.feats_head = nn.Sequential(
                nn.Linear(self.encoder_model.config.hidden_size, feats_out_label_size),
                nn.LeakyReLU()
            )

        self.pos_toggle = pos_toggle
        self.xpos_toggle = xpos_toggle
        self.lemma_toggle = lemma_toggle
        self.feats_toggle = feats_toggle

        # TODO add support back
        # self.deps_head = BiaffineParser(
        #     self.encoder_model.config.hidden_size, udtube_dropout, deprel_out_label_size
        # )

        # retrieving the LabelEncoders set up by the Dataset
        self.lemma_encoder = joblib.load(f"{self.path_name}/lemma_encoder.joblib")
        self.feats_encoder = joblib.load(f"{self.path_name}/ufeats_encoder.joblib")
        self.pos_encoder = joblib.load(f"{self.path_name}/upos_encoder.joblib")
        self.xpos_encoder = joblib.load(f"{self.path_name}/xpos_encoder.joblib")

        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        self.save_hyperparameters()
        self.dummy_tensor = torch.zeros(self.encoder_model.config.hidden_size, device=self.device)
        if checkpoint:
            checkpoint = torch.load(checkpoint)
            print("Loading from checkpoint")
            self.load_state_dict(checkpoint['state_dict'])

    def _load_model(self, model_name):
        if 'bert' in model_name:
            model = transformers.AutoModel.from_pretrained(
                model_name, output_hidden_states=True,
                hidden_dropout_prob=self.encoder_dropout
            )
        if 't5' in model_name:
            model = transformers.AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                dropout_rate=self.encoder_dropout
            )
            model = model.encoder

        # freezing params for first epoch
        for p in model.parameters():
            p.requires_grad = False
        print("Encoder Parameters frozen for the first Epoch")
        return model

    def _validate_input(
            self, pos_out_label_size, lemma_out_label_size, feats_out_label_size
    ):
        if (
                pos_out_label_size == 0
                or lemma_out_label_size == 0
                or feats_out_label_size == 0
        ):
            raise ValueError(
                "One of the label sizes given to the model was zero"
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
            if not torch.is_tensor(s):
                s = tensor(s, device=self.device)
            if len(s) != max_len:
                # I think the below operation puts the padding back on CPU, not great
                r_padding = torch.stack([pad] * (max_len - len(s)))
                r_padding = r_padding.to(s.device)
                padded_seq.append(torch.cat((s, r_padding)))
            else:
                padded_seq.append(s)
        if return_long:
            return torch.stack(padded_seq).long()
        return torch.stack(padded_seq)

    def pool_embeddings(
            self, x_embs: torch.tensor, tokenized: transformers.BatchEncoding, gold_label_ex=None
    ):
        new_embs, new_masks, words = [], [], []
        encodings = tokenized.encodings
        offset = 1
        if not encodings:
            encodings = tokenized.custom_encodings
            offset = 0 # no offset here, this is the Byt5 case
        for encoding, x_emb_i in zip(encodings, x_embs):
            embs_i, words_i, mask_i = [], [], []
            last_word_idx = slice(0, 0)
            # skipping over the first padding token ([CLS])
            for word_id, x_emb_j in zip(encoding.word_ids[offset:], x_emb_i[offset:]):
                if word_id is None:
                    # emb padding
                    break
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    word_emb_pooled = torch.mean(
                        x_emb_i[word_idxs], keepdim=True, dim=0
                    ).squeeze()
                    embs_i.append(word_emb_pooled)
                    try:
                        words_i.append("".join(encoding.tokens[word_idxs]).replace('##', ''))
                    except TypeError:
                        words_i.append(bytes(encoding.tokens[word_idxs]).decode())
                    mask_i.append(1)
            new_embs.append(torch.stack(embs_i))
            words.append(words_i[1:]) # ROOT is always first, dropping it here.
            new_masks.append(tensor(mask_i, device=self.device))
        if gold_label_ex:
            longest_seq = max(max(len(m) for m in words),
                            max(len(l) for l in gold_label_ex))
        else:
            longest_seq = max(len(m) for m in words)
        # longest_seq + 1 is the ROOT adjustment
        new_embs = self.pad_seq(new_embs, self.dummy_tensor, longest_seq + 1)
        new_masks = self.pad_seq(new_masks, tensor(0, device=self.device), longest_seq + 1)
        return new_embs, words, new_masks, longest_seq

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        grouped_params = [
            # {'params': self.deps_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.lemma_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.pos_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.xpos_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.feats_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.encoder_model.parameters(), 'lr': self.encoder_model_learning_rate, "weight_decay": 0.01}
        ]
        optimizer = torch.optim.AdamW(grouped_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def log_metrics(
            self,
            y_pred: torch.tensor,
            y_true: torch.tensor,
            batch_size: int,
            task_name: str,
            subset: str = "train",
            ignore_index: int = 0
    ):
        num_classes = y_pred.shape[1]
        # getting the pad
        acc = multiclass_accuracy(y_pred.permute(2, 1, 0), y_true.T, num_classes=num_classes,
                                  ignore_index=ignore_index, average="micro")
        self.log(
            f"{subset}_{task_name}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def _call_head(
            self,
            y_gold,
            logits,
            task_name="pos",
            subset="train",
            return_loss=True,
    ):
        batch_size, longest_seq, num_classes = logits.shape
        # ignore_idx is used for padding!
        lencoder = getattr(self, f"{task_name}_encoder")
        ignore_idx = int(lencoder.transform(["[PAD]"])[0]) # TODO better way?

        pad = tensor(ignore_idx, device=self.device)
        y_gold_tensor = self.pad_seq(
            y_gold, pad, longest_seq, return_long=True
        )

        # Each head returns ( batch X sequence_len X classes )
        # but CE & metrics want ( minibatch X Classes X sequence_len ); (minibatch, C, d0...dk) & (N, C, ..) in the docs
        logits = logits.permute(0, 2, 1)

        self.log_metrics(
            logits, y_gold_tensor, batch_size, task_name, subset=subset, ignore_index=ignore_idx
        )

        if return_loss:
            loss = nn.functional.cross_entropy(logits, y_gold_tensor, ignore_index=ignore_idx)
            return loss

    def _get_loss_from_deps_head(self, S_arc, S_lab, batch, attn_masks, subset="train"):
        batch_size = len(batch)

        # Removing root
        S_arc = S_arc[:, 1:, 1:]
        S_lab = S_lab[:, :, 1:, 1:]
        # S_lab_ignore_idx = S_lab.shape[1] - 1
        S_lab_ignore_idx = None
        attn_masks = attn_masks[:, 1:]
        longest_seq = S_arc.shape[1]

        # for head prediction, the padding is the length of the sequence, since num classes changes per sentence.
        gold_heads = self.pad_seq(
            batch.heads, tensor(longest_seq, device=self.device), longest_seq, return_long=True
        )
        gold_deprels = self.pad_seq(
            batch.deprels, tensor(S_lab_ignore_idx, device=self.device), longest_seq, return_long=True
        )
        # A lot of tensor manipulation goes into making the S_lab tenable.
        # Credit: https://github.com/daandouwe/biaffine-dependency-parser/tree/9338c6fde6de5393ac1bbdd6a8bb152c2a015a6c
        heads = gold_heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)  # [batch, n_labels, 1, sent_len]
        heads = torch.where(heads != longest_seq, heads, 0)  # Replacing padding due to index error
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)  # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)  # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
        labels = gold_deprels.view(-1)  # [batch*sent_len]

        srel_acc = multiclass_accuracy(S_lab, labels, num_classes=S_lab.shape[1], ignore_index=S_lab_ignore_idx, average="micro")
        self.log(
            f"{subset}_deprel_acc",
            srel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        # getting losses
        S_arc = S_arc * attn_masks.unsqueeze(dim=1)
        arc_loss = nn.functional.cross_entropy(S_arc, gold_heads, ignore_index=longest_seq)
        rel_loss = nn.functional.cross_entropy(S_lab, labels, ignore_index=S_lab_ignore_idx)

        return arc_loss, rel_loss

    def _decode_to_str(self, sentences, words, y_pos_logits, y_xpos_logits, y_lemma_logits,
                       y_feats_logits, replacements=None):
        # argmaxing
        if self.pos_toggle:
            y_pos_hat = torch.argmax(y_pos_logits, dim=-1)
        if self.xpos_toggle:
            y_xpos_hat = torch.argmax(y_xpos_logits, dim=-1)
        if self.lemma_toggle:
            y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        if self.feats_toggle:
            y_feats_hat = torch.argmax(y_feats_logits, dim=-1)

        # transforming to str
        y_pos_str_batch, y_xpos_str_batch, y_lemma_str_batch, y_feats_hat_batch = [], [], [], []
        for batch_idx, w_i in enumerate(words):
            lemmas_i = []
            seq_len = len(w_i)  # used to get rid of padding

            if self.pos_toggle:
                y_pos_str_batch.append(self.pos_encoder.inverse_transform(y_pos_hat[batch_idx][:seq_len].cpu()))
            if self.xpos_toggle:
                y_xpos_str_batch.append(self.xpos_encoder.inverse_transform(y_xpos_hat[batch_idx][:seq_len].cpu()))
            if self.feats_toggle:
                y_feats_hat_batch.append(self.feats_encoder.inverse_transform(y_feats_hat[batch_idx][:seq_len].cpu()))

            if self.lemma_toggle:
                # lemmas work through rule classification, so we have to also apply the rules.
                lemma_rules = self.lemma_encoder.inverse_transform(y_lemma_hat[batch_idx][:seq_len].cpu())
                for i, rule in enumerate(lemma_rules):
                    rscript = self.e_script.fromtag(rule)
                    lemma = rscript.apply(words[batch_idx][i])
                    lemmas_i.append(lemma)
                y_lemma_str_batch.append(lemmas_i)

        return sentences, words, y_pos_str_batch, y_xpos_str_batch, y_lemma_str_batch, y_feats_hat_batch, replacements

    def forward(self, batch: Union[TextBatch, ConlluBatch]):
        # getting raw embeddings
        x = self.encoder_model(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        # stacking n (self.pooling_layer) embedding layers
        x = torch.stack(
            x.hidden_states[-self.pooling_layers:]
        )
        # averages n layers into one embedding layer
        x = torch.mean(x, keepdim=True, dim=0).squeeze()

        # this is used for padding, when present
        if isinstance(batch, TextBatch):
            gold_label_ex = None
        else:
            gold_label_ex = batch.pos

        # converting x embeddings to the word level
        x, words, masks, longest_seq = self.pool_embeddings(
            x, batch.tokens, gold_label_ex=gold_label_ex
        )

        x_with_root = x.detach().clone()
        x = x[:, 1:, :]  # dropping the root token embedding for every task but dependency

        y_pos_logits = self.pos_head(x) if self.pos_toggle else None
        y_xpos_logits = self.xpos_head(x) if self.xpos_toggle else None
        y_lemma_logits = self.lemma_head(x) if self.lemma_toggle else None
        y_feats_logits = self.feats_head(x) if self.feats_toggle else None
        # S_arc, S_lab = self.deps_head(x_with_root)

        return batch.sentences, words, y_pos_logits, y_xpos_logits, y_lemma_logits, y_feats_logits, masks

    def training_step(
            self, batch: ConlluBatch, batch_idx: int, subset: str = "train"
    ):
        response = {}
        batch_size = len(batch)
        sentences, words, y_pos_logits, y_xpos_logits, y_lemma_logits, y_feats_logits, masks = self(batch)

        if self.pos_toggle:
            pos_loss = self._call_head(
                batch.pos,
                y_pos_logits,
                task_name="pos",
                subset=subset,
            )
            response["pos_loss"] = pos_loss

        if self.xpos_toggle:
            xpos_loss = self._call_head(
                batch.xpos,
                y_xpos_logits,
                task_name="xpos",
                subset=subset
            )
            response["xpos_loss"] = xpos_loss

        if self.lemma_toggle:
            lemma_loss = self._call_head(
                batch.lemmas,
                y_lemma_logits,
                task_name="lemma",
                subset=subset,
            )
            response["lemma_loss"] = lemma_loss

        if self.feats_toggle:
            feats_loss = self._call_head(
                batch.feats,
                y_feats_logits,
                task_name="feats",
                subset=subset,
            )
            response["feats_loss"] = feats_loss

        # TODO add back
        # getting loss from dep head, it's different from the rest
        # arc_loss, rel_loss = self._get_loss_from_deps_head(
        #     S_arc, S_lab,
        #     batch,
        #     masks,
        #     subset=subset)
        # response["arc_loss"] = arc_loss
        # response["rel_loss"] = rel_loss

        # combining the loss of the active heads
        loss = torch.mean(torch.stack([l for l in response.values()]))
        response["loss"] = loss

        self.log(
            f"{subset}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return response

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            for p in self.encoder_model.parameters():
                p.requires_grad = True
            print("Encoder Parameters unfrozen")

    def validation_step(self, batch: ConlluBatch, batch_idx: int):
        return self.training_step(batch, batch_idx, subset="val")

    def test_step(self, batch: ConlluBatch, batch_idx: int):
        sentences, words, y_pos_logits, y_xpos_logits, y_lemma_logits, y_feats_logits, masks = self(batch)

        # these calls log accuracy
        self._call_head(batch.pos, y_pos_logits, task_name="pos", subset="test", return_loss=False)
        self._call_head(batch.xpos, y_xpos_logits, task_name="xpos", subset="test", return_loss=False)
        self._call_head(batch.lemmas, y_lemma_logits, task_name="lemma", subset="test", return_loss=False)
        self._call_head(batch.feats, y_feats_logits, task_name="feats", subset="test", return_loss=False)

        return self._decode_to_str(sentences, words, y_pos_logits, y_xpos_logits, y_lemma_logits,
                                   y_feats_logits, replacements=batch.replacements)

    def predict_step(self, batch: TextBatch, batch_idx: int):
        *res, _ = self(batch)
        return self._decode_to_str(*res, replacements=batch.replacements)


if __name__ == "__main__":
    UDTubeCLI(UDTube, ConlluDataModule, save_config_callback=None)
