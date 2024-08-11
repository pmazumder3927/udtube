"""Models and CLI."""

import logging
from typing import Iterable, Optional, Callable, Any, Dict, Tuple, List, Union

import joblib
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.functional import classification
import transformers

# FIXME
from lightning.pytorch.core.optimizer import LightningOptimizer

from . import batches, defaults, edit_scripts


class UDTube(pl.LightningModule):
    """UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    classification using a single shared, pre-trained, tuned BERT-style
    encoder.

    Args:
        model_dir: model_directory.
        model_name: name of the model.
        pos_out_label_size: number of POS labels; usually provided by the
            dataset object.
        xpos_out_label_size: number of language-specific POS labels; usually
            provided by the dataset object.
        lemma_out_label_size: number of lemma tags; usually provided by the
            dataset.
        feats_out_label_size: number of feature tags; usually passed by the
            dataset.
        encoder_learning_rate: learning rate for the encoder layers.
        classifiers_learning_rate: learning rate for the classifier layers.
        pooling_layers: number of encoder layers to use to compute the
            embedding.
        train_from: path to checkpoint used to resume training.
        use_pos: if True, use POS tags.
        use_xpos: if True, use language-specific POS tags.
        use_lemma: if True, use lemmatization.
        use_feats: if True, use morphological feature tags.
    """

    # TODO: use defaults.

    def __init__(
        self,
        model_dir: str,
        train_from: Optional[str] = None,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        encoder: str = defaults.ENCODER,
        encoder_dropout: float = defaults.ENCODER_DROPOUT,
        encoder_learning_rate: float = defaults.ENCODER_LEARNING_RATE,
        pooling_layers: int = defaults.POOLING_LAYERS,
        classifiers_dropout: float = defaults.CLASSIFIERS_DROPOUT,
        classifiers_learning_rate: float = defaults.CLASSIFIERS_LEARNING_RATE,
        warmup_steps: int = defaults.WARMUP_STEPS,
        use_pos: bool = defaults.USE_POS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        # `2` is a dummy value here; it will be set by the data set object.
        pos_out_label_size: int = 2,
        xpos_out_label_size: int = 2,
        lemma_out_label_size: int = 2,
        feats_out_label_size: int = 2,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.classifiers_learning_rate = classifiers_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_dropout = encoder_dropout
        self.warmup_steps = warmup_steps
        self.step_adjustment = None
        self.pooling_layers = pooling_layers
        # FIXME this is wrong.
        self.encoder_model = self._load_model(model_name)
        self.dropout_layer = nn.Dropout(classifiers_dropout)
        # TODO: make heads nullable.
        if use_pos:
            self.pos_head = nn.Sequential(
                nn.Linear(
                    self.encoder_model.config.hidden_size, pos_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_xpos:
            self.xpos_head = nn.Sequential(
                nn.Linear(
                    self.encoder_model.config.hidden_size, xpos_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_lemma:
            self.lemma_head = nn.Sequential(
                nn.Linear(
                    self.encoder_model.config.hidden_size, lemma_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_feats:
            self.feats_head = nn.Sequential(
                nn.Linear(
                    self.encoder_model.config.hidden_size, feats_out_label_size
                ),
                nn.LeakyReLU(),
            )
        self.use_pos = use_pos
        self.use_xpos = use_xpos
        self.use_lemma = use_lemma
        self.use_feats = use_feats
        # Retrieves the LabelEncoders set up by the data set object.
        self.lemma_encoder = joblib.load(
            f"{self.model_dir}/lemma_encoder.joblib"
        )
        self.feats_encoder = joblib.load(
            f"{self.model_dir}/ufeats_encoder.joblib"
        )
        self.pos_encoder = joblib.load(f"{self.model_dir}/upos_encoder.joblib")
        self.xpos_encoder = joblib.load(
            f"{self.model_dir}/xpos_encoder.joblib"
        )
        self.edit_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        self.save_hyperparameters()
        self.dummy_tensor = torch.zeros(
            self.encoder_model.config.hidden_size, device=self.device
        )
        if train_from:
            logging.info("Loading from checkpoint %s", train_from)
            self.load_state_dict(torch.load(train_from)["state_dict"])

    # TODO: docs.

    def pad_seq(
        self,
        sequence: Iterable,
        pad: Iterable,
        max_len: int,
        return_long: bool = False,
    ) -> torch.tensor:
        # TODO find a way to keep this on GPU
        padded_seq = []
        for s in sequence:
            if not torch.is_tensor(s):
                s = torch.tensor(s, device=self.device)
            if len(s) != max_len:
                # TODO: I think the below operation puts padding back on CPU.
                r_padding = torch.stack([pad] * (max_len - len(s)))
                r_padding = r_padding.to(s.device)
                padded_seq.append(torch.cat((s, r_padding)))
            else:
                padded_seq.append(s)
        if return_long:
            return torch.stack(padded_seq).long()
        return torch.stack(padded_seq)

    # TODO: docs.

    def pool_embeddings(
        self,
        x_embs: torch.tensor,
        tokenized: transformers.BatchEncoding,
        gold_label_ex=None,
    ) -> Tuple[torch.tensor, List[str], torch.tensor, int]:
        new_embs = []
        new_masks = []
        words = []
        encodings = tokenized.encodings
        offset = 1
        if not encodings:
            encodings = tokenized.custom_encodings
            offset = 0  # no offset here, this is the Byt5 case
        for encoding, x_emb_i in zip(encodings, x_embs):
            embs_i = []
            words_i = []
            mask_i = []
            last_word_idx = slice(0, 0)
            # skipping over the first padding token ([CLS])
            for word_id, x_emb_j in zip(
                encoding.word_ids[offset:], x_emb_i[offset:]
            ):
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
                        words_i.append(
                            "".join(encoding.tokens[word_idxs])
                            .replace("##", "")
                            .replace("▁", "")
                        )
                    except TypeError:
                        words_i.append(
                            bytes(encoding.tokens[word_idxs]).decode()
                        )
                    mask_i.append(1)
            words.append(words_i)
            new_embs.append(torch.stack(embs_i))
            new_masks.append(torch.tensor(mask_i, device=self.device))
        if gold_label_ex:
            # TODO: variable names and remove noqa.
            longest_seq = max(
                max(len(m) for m in words),  # noqa: E741
                max(len(l) for l in gold_label_ex),  # noqa: E741
            )
        else:
            # TODO: variable names and remove noqa.
            longest_seq = max(len(m) for m in words)  # noqa: E741
        new_embs = self.pad_seq(new_embs, self.dummy_tensor, longest_seq)
        new_masks = self.pad_seq(
            new_masks, torch.tensor(0, device=self.device), longest_seq
        )
        return new_embs, words, new_masks, longest_seq

    # TODO: add additional optimizers.

    def configure_optimizers(self):
        """Prepare optimizer and schedulers."""
        grouped_params = [
            {
                "params": self.encoder_model.parameters(),
                "lr": self.encoder_learning_rate,
                "weight_decay": 0.01,
            }
        ]
        if self.use_lemma:
            grouped_params.append(
                {
                    "params": self.lemma_head.parameters(),
                    "lr": self.classifiers_learning_rate,
                }
            )
        if self.use_pos:
            grouped_params.append(
                {
                    "params": self.pos_head.parameters(),
                    "lr": self.classifiers_learning_rate,
                }
            )
        if self.use_xpos:
            grouped_params.append(
                {
                    "params": self.xpos_head.parameters(),
                    "lr": self.classifiers_learning_rate,
                }
            )
        if self.use_feats:
            grouped_params.append(
                {
                    "params": self.feats_head.parameters(),
                    "lr": self.classifiers_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(grouped_params)
        return [optimizer]

    # TODO: docs.

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[optim.Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if epoch > 0:
            # this only happens once params are unfrozen
            unfrozen_steps = self.trainer.global_step - self.step_adjustment
            if unfrozen_steps < self.warmup_steps:
                # Warming up LR from 0 to the encoder LR begins before the
                # encoder is unfrozen, so LR is not really at zero.
                lr_scale = 0 + float(unfrozen_steps) / self.warmup_steps
            else:
                # Decaying weight.
                lr_scale = 1 / (unfrozen_steps**0.5)
            optimizer.param_groups[0]["lr"] = (
                lr_scale * self.encoder_learning_rate
            )
            self.log(
                "encoder_lr",
                optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    # TODO: docs.

    def log_metrics(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        task_name: str,
        subset: str = "train",
        ignore_index: int = 0,
    ) -> None:
        num_classes = y_pred.shape[1]
        # getting the pad
        accuracy = classification.multiclass_accuracy(
            y_pred.permute(2, 1, 0),
            y_true.T,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="micro",
        )
        self.log(
            f"{subset}_{task_name}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    # TODO: docs.

    def _call_head(
        self,
        y_gold,
        logits,
        task_name="pos",
        subset="train",
        return_loss=True,
    ):
        batch_size, longest_seq, num_classes = logits.shape
        # ignore_idx is used for padding.
        lencoder = getattr(self, f"{task_name}_encoder")
        # TODO: better way?
        ignore_idx = int(lencoder.transform(["[PAD]"])[0])
        pad = torch.tensor(ignore_idx, device=self.device)
        y_gold_tensor = self.pad_seq(
            y_gold, pad, longest_seq, return_long=True
        )
        # Each head returns batch X sequence_len X classes, but we want
        # minibatch X classes X sequence_len.
        logits = logits.permute(0, 2, 1)
        self.log_metrics(
            logits,
            y_gold_tensor,
            batch_size,
            task_name,
            subset=subset,
            ignore_index=ignore_idx,
        )
        if return_loss:
            return nn.functional.cross_entropy(
                logits, y_gold_tensor, ignore_index=ignore_idx
            )

    # TODO: docs.

    def _decode_to_str(
        self,
        sentences,
        words,
        y_pos_logits,
        y_xpos_logits,
        y_lemma_logits,
        y_feats_logits,
        replacements=None,
    ) -> Tuple[str]:
        # argmaxing.
        if self.use_pos:
            y_pos_hat = torch.argmax(y_pos_logits, dim=-1)
        if self.use_xpos:
            y_xpos_hat = torch.argmax(y_xpos_logits, dim=-1)
        if self.use_lemma:
            y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        if self.use_feats:
            y_feats_hat = torch.argmax(y_feats_logits, dim=-1)
        # Transforms to string.
        y_pos_str_batch = []
        y_xpos_str_batch = []
        y_lemma_str_batch = []
        y_feats_hat_batch = []
        for batch_idx, w_i in enumerate(words):
            lemmas_i = []
            seq_len = len(w_i)  # Used to get rid of padding.
            if self.use_pos:
                y_pos_str_batch.append(
                    self.pos_encoder.inverse_transform(
                        y_pos_hat[batch_idx][:seq_len].cpu()
                    )
                )
            if self.use_xpos:
                y_xpos_str_batch.append(
                    self.xpos_encoder.inverse_transform(
                        y_xpos_hat[batch_idx][:seq_len].cpu()
                    )
                )
            if self.use_feats:
                y_feats_hat_batch.append(
                    self.feats_encoder.inverse_transform(
                        y_feats_hat[batch_idx][:seq_len].cpu()
                    )
                )
            if self.use_lemma:
                # One has to apply the rule first.
                lemma_rules = self.lemma_encoder.inverse_transform(
                    y_lemma_hat[batch_idx][:seq_len].cpu()
                )
                for i, rule in enumerate(lemma_rules):
                    rscript = self.edit_script.fromtag(rule)
                    lemma = rscript.apply(words[batch_idx][i])
                    lemmas_i.append(lemma)
                y_lemma_str_batch.append(lemmas_i)
        return (
            sentences,
            words,
            y_pos_str_batch,
            y_xpos_str_batch,
            y_lemma_str_batch,
            y_feats_hat_batch,
            replacements,
        )

    # TODO: docs.

    def forward(
        self, batch: batches.CustomBatch
    ) -> Tuple[Iterable[str], list[list[str]], torch.Tensor]:
        # If something is longer than an allowed sequence, we trim it down.
        max_length = self.encoder_model.config.max_position_embeddings
        actual_length = batch.tokens.input_ids.shape[1]
        if actual_length > max_length:
            logging.warning(
                "Truncating sequence from %d to %d", actual_length, max_length
            )
            batch.tokens.input_ids = batch.tokens.input_ids[:max_length]
            batch.tokens.attention_mask = batch.tokens.attention_mask[
                :max_length
            ]
        # Gets raw embeddings.
        x = self.encoder_model(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        # Stacks the pooling layers.
        x = torch.stack(x.hidden_states[-self.pooling_layers :])
        # Averages them into one embedding layer.
        x = torch.mean(x, keepdim=True, dim=0).squeeze()
        # TODO: explain this.
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)
        # Applies dropout.
        x = self.dropout_layer(x)
        # TODO: explain this.
        if isinstance(batch, batches.TextBatch):
            gold_label_ex = None
        else:
            gold_label_ex = batch.pos
        # Maps from subword embeddings to word-level embeddings.
        x, words, masks, longest_seq = self.pool_embeddings(
            x, batch.tokens, gold_label_ex=gold_label_ex
        )
        y_pos_logits = self.pos_head(x) if self.use_pos else None
        y_xpos_logits = self.xpos_head(x) if self.use_xpos else None
        y_lemma_logits = self.lemma_head(x) if self.use_lemma else None
        y_feats_logits = self.feats_head(x) if self.use_feats else None
        # TODO(#6): make the response an object.
        return (
            batch.sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        )

    # TODO: docs.

    def training_step(
        self, batch: batches.ConlluBatch, batch_idx: int, subset: str = "train"
    ) -> Dict[str, float]:
        losses = {}
        (
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        ) = self(batch)
        if self.use_pos:
            pos_loss = self._call_head(
                batch.pos,
                y_pos_logits,
                task_name="pos",
                subset=subset,
            )
            losses["pos_loss"] = pos_loss
        if self.use_xpos:
            xpos_loss = self._call_head(
                batch.xpos, y_xpos_logits, task_name="xpos", subset=subset
            )
            losses["xpos_loss"] = xpos_loss
        if self.use_lemma:
            lemma_loss = self._call_head(
                batch.lemmas,
                y_lemma_logits,
                task_name="lemma",
                subset=subset,
            )
            losses["lemma_loss"] = lemma_loss
        if self.use_feats:
            feats_loss = self._call_head(
                batch.feats,
                y_feats_logits,
                task_name="feats",
                subset=subset,
            )
            losses["feats_loss"] = feats_loss
        # Averages the loss of all active heads.
        losses["loss"] = loss = torch.mean(torch.stack(list(losses.values())))
        self.log(
            f"{subset}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    # TODO: docs.

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            # How many steps are in an epoch?
            self.step_adjustment = self.trainer.global_step
            for p in self.encoder_model.parameters():
                p.requires_grad = True
            logging.info("Encoder Parameters unfrozen")

    # TODO: docs.

    def validation_step(
        self, batch: batches.ConlluBatch, batch_idx: int
    ) -> Dict[str, float]:
        return self.training_step(batch, batch_idx, subset="val")

    # TODO: docs.

    def test_step(
        self, batch: batches.ConlluBatch, batch_idx: int
    ) -> Tuple[str]:
        (
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        ) = self(batch)
        return self._decode_to_str(
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            replacements=batch.replacements,
        )

    # TODO: docs.

    def predict_step(
        self, batch: batches.TextBatch, batch_idx: int
    ) -> Tuple[str]:
        *res, _ = self(batch)
        return self._decode_to_str(*res, replacements=batch.replacements)
