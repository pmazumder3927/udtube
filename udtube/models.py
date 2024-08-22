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

from . import data, defaults


class UDTube(pl.LightningModule):
    """UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    classification using a single shared, pre-trained, tuned BERT-style
    encoder.

    Args:
        model_dir: Model directory.
        reverse_edits: If true, uses reverse (suffixal) edit scripts.
        language: ISO-639-1 language code, for the UDPipe tokenizer.
        encoder: Name of the encoder from Hugging Face.
        encoder_learning_rate: Learning rate for the encoder layers.
        encoder_dropout: Dropout probability for the encoder layers.
        pooling_layers: Number of encoder layers to use to compute the
            embedding.
        classifier_learning_rate: Learning rate for the classifier layers.
        classifier_dropout: Dropout probability for the classifier layers.
        warmup_steps: Number of warmup steps.
        use_upos: If true, use POS tags.
        use_xpos: If true, use language-specific POS tags.
        use_lemma: If true, use lemmatization.
        use_feats: If true, use morphological feature tags.
        upos_out_size: Number of POS labels; usually provided by the
            dataset object.
        xpos_out_size: Number of language-specific POS labels; usually
            provided by the dataset object.
        lemma_out_size: Number of lemma tags; usually provided by the
            dataset.
        feats_out_size: Number of feature tags; usually passed by the
            dataset.
    """

    # Initialization.

    def __init__(
        self,
        model_dir: str,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        encoder: str = defaults.ENCODER,
        encoder_dropout: float = defaults.ENCODER_DROPOUT,
        encoder_learning_rate: float = defaults.ENCODER_LEARNING_RATE,
        pooling_layers: int = defaults.POOLING_LAYERS,
        classifier_dropout: float = defaults.CLASSIFIER_DROPOUT,
        classifier_learning_rate: float = defaults.CLASSIFIER_LEARNING_RATE,
        warmup_steps: int = defaults.WARMUP_STEPS,
        use_upos: bool = defaults.USE_POS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        # `2` is a dummy value here; it will be set by the data set object.
        upos_out_size: int = 2,
        xpos_out_size: int = 2,
        lemma_out_size: int = 2,
        feats_out_size: int = 2,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.classifier_learning_rate = classifier_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_dropout = encoder_dropout
        self.warmup_steps = warmup_steps
        self.step_adjustment = None
        self.pooling_layers = pooling_layers
        self.encoder_model = self._load_model(encoder)
        self.dropout_layer = nn.Dropout(classifier_dropout)
        self.upos_head = self._make_head(upos_out_size) if use_upos else None
        self.xpos_head = self._make_head(xpos_out_size) if use_xpos else None
        self.lemma_head = self._make_head(lemma_out_size) if use_lemma else None
        self.feats_head = self._make_head(feats_out_size) if use_feats else None
        self.save_hyperparameters()

    def _make_head(self, out_size: int) -> nn.Sequential:
        """Helper for generating heads.

        Args:
            out_size (int).
        
        Returns:
            A sequential linear layer.
        """
        return nn.Sequential(nn.Linear(self.hidden_size, out_size), nn.LeakyReLU())

    def group_embeddings(
        self,
        embeddings: torch.Tensor,
        tokenized: transformers.BatchEncoding,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """Groups subword embeddings to form word embeddings.

        This is necessary because each classifier head makes per-word
        decisions, but the contextual embeddings use subwords. Therefore,
        we average over subwords.

        Args:
            embeddings: the embeddings tensor to pool.
            tokens: the batch encoding.

        Returns:
            The re-pooled embeddings tensor and the words for the batch (as a
                list of lists).
        """
        embeddings_list = []
        words_list = []
        for encoding, x_emb_i in zip(tokenized.encodings, x_embeddings):
            embeddings_i = []
            words_i = []
            last_word_idx = slice(0, 0)
            for word_id, x_emb_j in zip(encoding.word_ids, x_emb_i):
                # Padding.
                if word_id is None:
                    break
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    word_emb_pooled = torch.mean(x_emb_i[word_idxs], dim=0)
                    embeddings_i.append(word_emb_pooled)
                    words_i.append(
                        "".join(encoding.tokens[word_idxs])
                         .replace("##", "")
                         .replace("▁", "")
                    )
            embeddings_list.append(torch.stack(embeddings_i))
            words_list.append(words_i)
        new_embeddings = data.pad_tensors(embeddings_list)
        return new_embeddings, words

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
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_upos:
            grouped_params.append(
                {
                    "params": self.upos_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_xpos:
            grouped_params.append(
                {
                    "params": self.xpos_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_feats:
            grouped_params.append(
                {
                    "params": self.feats_head.parameters(),
                    "lr": self.classifier_learning_rate,
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
            # This only happens once params are unfrozen
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
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        task_name: str,
        subset: str = "train",
        ignore_index: int = 0,
    ) -> None:
        accuracy = classification.multiclass_accuracy(
            y_pred.permute(2, 1, 0),
            y_true.T,
            num_classes=y_pred.shape[1],
            ignore_index=pad_idx,
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
        task_name: str,
        subset: str = "train",
        return_loss=True,
    ):
        # Each head returns batch X sequence_len X classes, but we want
        # batch X classes X sequence_len.
        logits = logits.permute(0, 2, 1)
        self.log_metrics(
            logits,
            y_gold,
            batch_size=len(logits),
            task_name,
            subset=subset,
            ignore_index=pad_idx,
        )
        if return_loss:
            return nn.functional.cross_entropy(
                logits, y_gold_tensor, ignore_index=pad_idx,
            )

    # TODO: docs.
    # TODO: this is not a good implementation.

    def _decode_to_str(
        self,
        sentences,
        words,
        y_upos_logits,
        y_xpos_logits,
        y_lemma_logits,
        y_feats_logits,
        replacements=None,
    ) -> Tuple[str]:
        # argmaxing.
        if self.use_upos:
            y_upos_hat = torch.argmax(y_upos_logits, dim=-1)
        if self.use_xpos:
            y_xpos_hat = torch.argmax(y_xpos_logits, dim=-1)
        if self.use_lemma:
            y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        if self.use_feats:
            y_feats_hat = torch.argmax(y_feats_logits, dim=-1)
        # Transforms to string.
        y_upos_str_batch = []
        y_xpos_str_batch = []
        y_lemma_str_batch = []
        y_feats_hat_batch = []
        for batch_idx, w_i in enumerate(words):
            lemmas_i = []
            seq_len = len(w_i)  # Used to get rid of padding.
            if self.use_upos:
                y_upos_str_batch.append(
                    self.upos_encoder.inverse_transform(
                        y_upos_hat[batch_idx][:seq_len].cpu()
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
            y_upos_str_batch,
            y_xpos_str_batch,
            y_lemma_str_batch,
            y_feats_hat_batch,
            replacements,
        )

    # TODO: docs.
    # TODO: typing.

    def forward(
        self,
        batch: data.Batch,
    ) -> Tuple[Iterable[str], List[List[str]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # If something is longer than an allowed sequence, we trim it down.
        actual_length = batch.tokens.input_ids.shape[1]
        max_length = self.encoder_model.config.max_position_embeddings
        if actual_length > max_length:
            logging.warning(
                "truncating sequence from %d to %d", actual_length, max_length
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
        # Maps from subword embeddings to word-level embeddings.
        x, words = self.repool_embeddings(x, batch.tokens)
        # Applies classification heads.
        y_upos_logits = self.upos_head(x) if self.use_upos else None
        y_xpos_logits = self.xpos_head(x) if self.use_xpos else None
        y_lemma_logits = self.lemma_head(x) if self.use_lemma else None
        y_feats_logits = self.feats_head(x) if self.use_feats else None
        # TODO(#6): make the response an object.
        return (
            batch.texts,
            words,
            y_upos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
        )

    # TODO: docs.

    def training_step(
        self, batch: batches.ConlluBatch, batch_idx: int, subset: str = "train"
    ) -> Dict[str, float]:
        losses = {}
        (
            _,
            _,
            y_upos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            _,
        ) = self(batch)
        if self.use_upos:
            pos_loss = self._call_head(
                batch.pos,
                y_upos_logits,
                task_name="upos",
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
            for parameter in self.encoder_model.parameters():
                parameter.requires_grad = True
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
            y_upos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            _,
        ) = self(batch)
        return self._decode_to_str(
            sentences,
            words,
            y_upos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
        )

    # TODO: docs.

    def predict_step(
        self, batch: batches.TextBatch, batch_idx: int
    ) -> Tuple[str]:
        *res, _ = self(batch)
        return self._decode_to_str(*res)

    # Properties.

    @property
    def hidden_size(self) -> int:
        return self.encoder_model.config.hidden_size

    @property
    def use_upos(self) -> bool:
        return self.upos_head is not None

    @property
    def use_xpos(self) -> bool:
        return self.xpos_head is not None

    @property
    def use_lemma(self) -> bool:
        return self.lemma_head is not None

    @property
    def use_feats(self) -> bool:
        return self.feats_head is not None
