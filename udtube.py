from typing import Iterable

import lightning.pytorch as pl
import torch
import transformers
import joblib
import edit_scripts
from lightning.pytorch.cli import LightningCLI
from torch import nn, tensor
from torchmetrics import Accuracy
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
            "data.lemma_classes_cnt",
            "model.lemma_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_classes_cnt",
            "model.feats_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.deprel_classes_cnt",
            "model.deprel_out_label_size",
            apply_on="instantiate"
        )
        parser.link_arguments(
            "data.tokenizer_size",
            "model.tokenizer_size",
            apply_on="instantiate"
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
            lemma_out_label_size: int = 2,
            feats_out_label_size: int = 2,
            deprel_out_label_size: int = 2,
            tokenizer_size: int = None,
            udtube_dropout: float = 0.3,
            encoder_dropout: float = 0.5,
            udtube_learning_rate: float = 5e-3,
            encoder_model_learning_rate: float = 2e-5,
            pooling_layers: int = 4,
            reverse_edits: bool = False,
            checkpoint: str = None
    ):
        """Initializes the instance based on user input.

        Args:
            path_name: The name of the folder to save/load model assets. This includes the lightning logs and the encoders.
            model_name: The name of the model; used to tokenize and encode.
            pos_out_label_size: The amount of POS labels. This is usually passed by the dataset.
            lemma_out_label_size: The amount of lemma rule labels in the dataset. This is usually passed by the dataset.
            feats_out_label_size: The amount of feature labels in the dataset. This is usually passed by the dataset.
            udtube_learning_rate: The learning rate of the full model
            encoder_model_learning_rate: The learning rate of only the encoder
            pooling_layers: The amount of layers used for embedding calculation
            checkpoint: The model checkpoint file
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
        # last item in the list is a pad from the label encoder
        self.pos_pad = tensor(
            pos_out_label_size - 1, device=self.device
        )
        self.lemma_pad = tensor(lemma_out_label_size - 1, device=self.device)
        self.feats_pad = tensor(feats_out_label_size - 1, device=self.device)
        self.deprel_pad = tensor(deprel_out_label_size - 1, device=self.device)
        self.encoder_model = self._load_model(model_name, tokenizer_size)
        self.pos_head = nn.Sequential(
            nn.Linear(self.encoder_model.config.hidden_size, pos_out_label_size),
            nn.Tanh(),
        )
        self.lemma_head = nn.Sequential(
            nn.Linear(self.encoder_model.config.hidden_size, lemma_out_label_size),
            nn.Tanh(),
        )
        self.feats_head = nn.Sequential(
            nn.Linear(self.encoder_model.config.hidden_size, feats_out_label_size),
            nn.Tanh(),
        )
        self.deps_head = BiaffineParser(
            self.encoder_model.config.hidden_size, udtube_dropout, deprel_out_label_size
        )
        # retrieving the LabelEncoders set up by the Dataset
        self.lemma_encoder = joblib.load(f"{self.path_name}/lemma_encoder.joblib")
        self.ufeats_encoder = joblib.load(f"{self.path_name}/ufeats_encoder.joblib")
        self.upos_encoder = joblib.load(f"{self.path_name}/upos_encoder.joblib")
        self.deprel_encoder = joblib.load(f"{self.path_name}/deprel_encoder.joblib")

        # Setting up all the metrics objects for each task
        self.pos_loss = nn.CrossEntropyLoss(
            ignore_index=pos_out_label_size - 1
        )
        self.pos_accuracy = Accuracy(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=pos_out_label_size - 1,
        )

        self.lemma_loss = nn.CrossEntropyLoss(
            ignore_index=lemma_out_label_size - 1
        )
        self.lemma_accuracy = Accuracy(
            task="multiclass",
            num_classes=lemma_out_label_size,
            ignore_index=lemma_out_label_size - 1,
        )

        self.feats_loss = nn.CrossEntropyLoss(
            ignore_index=feats_out_label_size - 1
        )
        self.feats_accuracy = Accuracy(
            task="multiclass",
            num_classes=feats_out_label_size,
            ignore_index=feats_out_label_size - 1,
        )

        self.deprel_accuracy = Accuracy(
            task="multiclass",
            num_classes=deprel_out_label_size,
            ignore_index=deprel_out_label_size - 1,
        )
        self.deprel_loss = nn.CrossEntropyLoss(
            ignore_index=deprel_out_label_size - 1
        )

        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        self.save_hyperparameters()
        self.dummy_tensor = torch.zeros(self.encoder_model.config.hidden_size, device=self.device)
        if checkpoint:
            self.load_state_dict(checkpoint['state_dict'])

    def _load_model(self, model_name, tokenizer_size):
        model = transformers.AutoModel.from_pretrained(
            model_name, output_hidden_states=True,
            hidden_dropout_prob=self.encoder_dropout
        )
        if 't5' in model_name:
            model = model.encoder
        if tokenizer_size:
            model.resize_token_embeddings(tokenizer_size)
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
            self, x_embs: torch.tensor, tokenized: transformers.BatchEncoding, gold_label_ex
    ):
        new_embs = []
        new_masks = []
        words = []
        for encoding, x_emb_i in zip(tokenized.encodings, x_embs):
            embs_i = []
            words_i = []
            mask_i = []
            last_word_idx = slice(0, 0)
            # skipping over the first padding token ([CLS])
            for word_id, x_emb_j in zip(encoding.word_ids[1:], x_emb_i[1:]):
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
                    words_i.append("".join(encoding.tokens[word_idxs]).replace('##', ''))
                    mask_i.append(1)
            new_embs.append(torch.stack(embs_i))
            words.append(words_i)
            new_masks.append(tensor(mask_i, device=self.device))
        longest_seq = max(max(len(m) for m in new_masks),
                          max(len(l) for l in gold_label_ex))  # while seq mismatches exist
        new_embs = self.pad_seq(new_embs, self.dummy_tensor, longest_seq)
        new_masks = self.pad_seq(new_masks, tensor(0, device=self.device), longest_seq)
        return new_embs, words, new_masks, longest_seq

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        grouped_params = [
            {'params': self.deps_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.lemma_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.pos_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.feats_head.parameters(), 'lr': self.udtube_learning_rate},
            {'params': self.encoder_model.parameters(), 'lr': self.encoder_model_learning_rate, "weight_decay": 0.01}
        ]
        optimizer = torch.optim.AdamW(grouped_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def log_metrics(
            self,
            y_pred: torch.tensor,
            y_true: torch.tensor,
            batch_size: int,
            task_name: str,
            subset: str = "train",
    ):
        accuracy = getattr(self, f"{task_name}_accuracy")
        self.log(
            f"{subset}:{task_name}_acc",
            accuracy(y_pred, y_true),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def _get_loss_from_head(
            self,
            y_gold,
            longest_seq,
            x_word_embs,
            batch_size,
            task_name="pos",
            subset="train",
    ):
        pad = getattr(self, f"{task_name}_pad")
        head = getattr(self, f"{task_name}_head")
        obj_func = getattr(self, f"{task_name}_loss")

        # need to do some preprocessing on Y
        # Has to be done here, after the adjustment of x_embs to the word level
        y_gold_tensor = self.pad_seq(
            y_gold, pad, longest_seq, return_long=True
        )

        # getting logits from head, and then permuting them for metrics calculation
        # Each head returns ( batch X sequence_len X classes )
        # but CE & metrics want ( minibatch X Classes X sequence_len ); (minibatch, C, d0...dk) & (N, C, ..) in the docs
        logits = head(x_word_embs)
        logits = logits.permute(0, 2, 1)

        # getting loss and logging
        loss = obj_func(logits, y_gold_tensor)
        self.log_metrics(
            logits, y_gold_tensor, batch_size, task_name, subset=subset
        )
        return loss

    def _get_loss_from_deps_head(self, x_word_embs, batch, longest_seq, attn_masks, subset="train"):
        S_arc, S_lab = self.deps_head(x_word_embs)
        batch_size = len(batch)

        # for head prediction, the padding is the length of the sequence, since num classes changes per sentence.
        gold_heads = self.pad_seq(
            batch.heads, tensor(longest_seq, device=self.device), longest_seq, return_long=True
        )
        gold_deprels = self.pad_seq(
            batch.deprels, self.deprel_pad, longest_seq, return_long=True
        )
        # this is not an acc object, since head has a different amount of labels per sequence (L dim = S dim)
        self.log(
            f"{subset}:head_acc",
            multiclass_accuracy(S_arc, gold_heads, longest_seq, ignore_index=longest_seq),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
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
        self.log(
            f"{subset}:dep_rel_acc",
            self.deprel_accuracy(S_lab, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )
        # getting losses
        S_arc = S_arc * attn_masks.unsqueeze(dim=1)
        # longest_seq is the ignored index because it is the padding used for this task
        arc_loss = nn.functional.cross_entropy(S_arc, gold_heads, ignore_index=longest_seq)
        rel_loss = self.deprel_loss(S_lab, labels)

        return arc_loss, rel_loss


    def _decode_to_str(self, words, y_pos_logits, y_lemma_logits, y_feats_logits, S_arc_logits, S_rel_logits):
        # argmaxing
        y_pos_hat = torch.argmax(y_pos_logits, dim=-1)
        y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        y_feats_hat = torch.argmax(y_feats_logits, dim=-1)
        y_s_arc_hat = torch.argmax(S_arc_logits, dim=-1)
        y_s_rel_hat = torch.argmax(S_rel_logits, dim=-1)

        # transforming to str
        y_pos_str_batch = []
        y_lemma_str_batch = []
        y_feats_hat_batch = []
        y_s_arc_batch = []
        y_s_rel_batch = []
        for batch_idx in range(len(words)):
            lemmas_i = []
            y_pos_str_batch.append(self.upos_encoder.inverse_transform(y_pos_hat[batch_idx]))
            y_feats_hat_batch.append(self.ufeats_encoder.inverse_transform(y_feats_hat[batch_idx]))
            y_s_rel_batch.append(self.deprel_encoder.inverse_transform(y_s_arc_hat[batch_idx]))
            y_s_arc_batch.append(str(y_s_rel_hat[batch_idx])) # these were never encoded!

            # lemmas work through rule classification, so we have to also apply the rules.
            lemma_rules = self.lemma_encoder.inverse_transform(y_lemma_hat[batch_idx])
            for i, rule in enumerate(lemma_rules):
                rscript = self.e_script.fromtag(rule)
                lemma = rscript.apply(words[batch_idx][i])
                lemmas_i.append(lemma)
            y_lemma_str_batch.append(lemmas_i)

        return words, y_pos_str_batch, y_lemma_str_batch, y_feats_hat_batch, y_s_arc_batch, y_s_rel_batch

    def forward(self, batch: Union[TextBatch, ConlluBatch]):
        x_encoded = self.encoder_model(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_n_layer_embs = torch.stack(
            x_encoded.hidden_states[-self.pooling_layers:]
        )
        x_embs = torch.mean(last_n_layer_embs, keepdim=True, dim=0).squeeze()
        x_word_embs, words, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens
        )

        y_pos_logits = self.pos_head(x_word_embs)
        y_lemma_logits = self.lemma_head(x_word_embs)
        y_feats_logits = self.feats_head(x_word_embs)
        S_arc, S_lab = self.deps_head(x_word_embs)

        return words, y_pos_logits, y_lemma_logits, y_feats_logits, S_arc, S_lab

    def training_step(
            self, batch: ConlluBatch, batch_idx: int, subset: str = "train"
    ):
        x_encoded = self.encoder_model(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_n_layer_embs = torch.stack(
            x_encoded.hidden_states[-self.pooling_layers:]
        )
        batch_size = len(batch)
        x_embs = torch.mean(last_n_layer_embs, keepdim=True, dim=0).squeeze(dim=0)

        x_word_embs, words, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens, batch.pos
        )

        # TODO, delete this, using it to understand how often this happens
        for s, g in zip(words, batch.pos):
            if len(s) != len(g):
                print(
                    f"sequence length mismatch, s = {len(s)}, g = {len(g)}. Something in {words} is tokenized incorrectly.")

        # need to do some preprocessing on Y
        # Has to be done here, after the adjustment of x_embs to the word level
        pos_loss = self._get_loss_from_head(
            batch.pos,
            longest_seq,
            x_word_embs,
            batch_size,
            task_name="pos",
            subset=subset,
        )
        lemma_loss = self._get_loss_from_head(
            batch.lemmas,
            longest_seq,
            x_word_embs,
            batch_size,
            task_name="lemma",
            subset=subset,
        )
        feats_loss = self._get_loss_from_head(
            batch.feats,
            longest_seq,
            x_word_embs,
            batch_size,
            task_name="feats",
            subset=subset,
        )

        # getting loss from dep head, it's different from the rest
        arc_loss, rel_loss = self._get_loss_from_deps_head(
            x_word_embs,
            batch,
            longest_seq,
            attn_masks,
            subset=subset)

        # combining the loss of the heads
        loss = torch.mean(torch.stack([pos_loss, lemma_loss, feats_loss, arc_loss, rel_loss]))
        self.log(
            f"{subset}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return {"loss": loss}

    def validation_step(self, batch: ConlluBatch, batch_idx: int):
        return self.training_step(batch, batch_idx, subset="val")

    def test_step(self, batch: ConlluBatch, batch_idx: int):
        res = self(batch)
        return self._decode_to_str(*res)

    def predict_step(self, batch: TextBatch, batch_idx: int):
        res = self(batch)
        return self._decode_to_str(*res)


if __name__ == "__main__":
    UDTubeCLI(UDTube, ConlluDataModule)
