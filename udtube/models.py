"""The UDTube model.

In the documentation below, N is the batch size, C is the number of classes
for a classification head, and L is the maximum length (in subwords, tokens,
or tags) of a sentence in the batch.
"""

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.functional import classification

from . import data, defaults, modules, special


class UDTube(pl.LightningModule):
    """The UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    tagging using a single shared, pre-trained, fine-tuned BERT-style encoder
    and sequential linear classifiers for each subtask.
    """

    encoder: modules.UDTubeEncoder
    classifier: modules.UDTubeClassifier
    loss_func: nn.CrossEntropyLoss

    def __init__(
        self,
        encoder: str = defaults.ENCODER,
        dropout: float = defaults.DROPOUT,
        pooling_layers: int = defaults.POOLING_LAYERS,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        *,
        # `2` is a dummy value here; it will be set by the data set object.
        upos_out_size: int = 2,
        xpos_out_size: int = 2,
        lemma_out_size: int = 2,
        feats_out_size: int = 2,
        # Optimization and LR scheduling for the encoder.
        encoder_optimizer: str = defaults.OPTIMIZER,
        encoder_learning_rate: float = defaults.LEARNING_RATE,
        encoder_beta1: float = defaults.BETA1,
        encoder_beta2: float = defaults.BETA2,
        encoder_scheduler: str = defaults.SCHEDULER,
        encoder_reduceonplateau_factor: float = defaults.REDUCEONPLATEAU_FACTOR,  # noqa: E501
        encoder_reduceonplateau_patience: float = defaults.REDUCEONPLATEAU_PATIENCE,  # noqa: E501
        encoder_min_learning_rate: float = defaults.MIN_LEARNING_RATE,
        encoder_warmup_steps: int = defaults.WARMUP_STEPS,
        # Optimization and LR scheduling for the classifier.
        classifier_optimizer: str = defaults.OPTIMIZER,
        classifier_learning_rate: float = defaults.LEARNING_RATE,
        classifier_beta1: float = defaults.BETA1,
        classifier_beta2: float = defaults.BETA2,
        classifier_scheduler: str = defaults.SCHEDULER,
        classifier_reduceonplateau_factor: float = defaults.REDUCEONPLATEAU_FACTOR,  # noqa: E501
        classifier_reduceonplateau_patience: float = defaults.REDUCEONPLATEAU_PATIENCE,  # noqa: E501
        classifier_min_learning_rate: float = defaults.MIN_LEARNING_RATE,
        classifier_warmup_steps: int = defaults.WARMUP_STEPS,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = modules.UDTubeEncoder(
            encoder,
            dropout,
            pooling_layers,
            encoder_optimizer=encoder_optimizer,
            encoder_learning_rate=encoder_learning_rate,
            encoder_beta1=encoder_beta1,
            encoder_beta2=encoder_beta2,
            encoder_scheduler=encoder_scheduler,
            encoder_reduceonplateau_factor=encoder_reduceonplateau_factor,
            encoder_reduceonplateau_patience=encoder_reduceonplateau_patience,
            encoder_min_learning_rate=encoder_min_learning_rate,
            encoder_warmup_steps=encoder_warmup_steps,
        )
        self.classifier = modules.UDTubeClassifier(
            use_upos,
            use_xpos,
            use_lemma,
            use_feats,
            upos_out_size=upos_out_size,
            xpos_out_size=xpos_out_size,
            lemma_out_size=lemma_out_size,
            feats_out_size=feats_out_size,
            classifier_optimizer=classifier_optimizer,
            classifier_learning_rate=classifier_learning_rate,
            classifier_beta1=classifier_beta1,
            classifier_beta2=classifier_beta2,
            classifier_scheduler=classifier_scheduler,
            classifier_reduceonplateau_factor=classifier_reduceonplateau_factor,  # noqa: E501
            classifier_reduceonplateau_patience=classifier_reduceonplateau_patience,  # noqa: E501
            classifier_min_learning_rate=classifier_min_learning_rate,
            classifier_warmup_steps=classifier_warmup_steps,
        )
        # Other stuff.
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)
        self.save_hyperparameters()

    # Properties.

    @property
    def use_upos(self) -> bool:
        return self.classifier.use_upos

    @property
    def use_xpos(self) -> bool:
        return self.classifier.use_xpos

    @property
    def use_lemma(self) -> bool:
        return self.classifier.use_lemma

    @property
    def use_feats(self) -> bool:
        return self.classifier.use_feats

    # Forward pass.

    def forward(
        self,
        batch: data.Batch,
    ) -> modules.UDTubeLogits:
        return self.classifier(self.encoder(batch))

    # Required API.

    def configure_optimizers(
        self,
    ) -> Tuple[Dict, Dict]:
        """Prepare optimizer and schedulers."""
        return (
            self.encoder.configure_optimizers(),
            self.classifier.configure_optimizers(),
        )

    @staticmethod
    def _accuracy(logits: torch.Tensor, golds: torch.Tensor) -> float:
        """Computes multi-class micro-accuracy for a head.

        Args:
            logits: logits, of shape N x C x L.
            golds: gold data, of shape N x L.

        Returns:
            The multi-class micro-accuracy.
        """
        return classification.multiclass_accuracy(
            logits,
            golds,
            num_classes=logits.shape[1],
            average="micro",
            ignore_index=special.PAD_IDX,
        )

    # TODO: docs.

    def _log_accuracy(
        self,
        golds: torch.Tensor,
        logits: torch.Tensor,
        task_name: str,
        subset: str = "train",
    ) -> None:
        self.log(
            f"{subset}_{task_name}_accuracy",
            self._accuracy(logits, golds),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(logits),
        )

    # TODO: docs.

    def _inference_step(
        self, batch: data.ConlluBatch, batch_idx: int, subset: str
    ) -> None:
        logits = self(batch)
        # Computes and logs loss and accuracy.
        losses = []
        if self.use_upos:
            losses.append(self.loss_func(logits.upos, batch.upos))
            self._log_accuracy(
                batch.upos, logits.upos, task_name="upos", subset=subset
            )
        if self.use_xpos:
            losses.append(self.loss_func(logits.xpos, batch.xpos))
            self._log_accuracy(
                batch.xpos, logits.xpos, task_name="xpos", subset=subset
            )
        if self.use_lemma:
            losses.append(self.loss_func(logits.lemma, batch.lemma))
            self._log_accuracy(
                batch.lemma,
                logits.lemma,
                task_name="lemma",
                subset=subset,
            )
        if self.use_feats:
            losses.append(self.loss_func(logits.feats, batch.feats))
            self._log_accuracy(
                batch.feats,
                logits.feats,
                task_name="feats",
                subset=subset,
            )
        loss = torch.mean(torch.stack(losses))
        self.log(
            f"{subset}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    def training_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
        subset: str = "train",
    ) -> None:
        """Runs the forward pass, logs, and steps the optimizers."""
        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss = self._inference_step(batch, batch_idx, subset)
        self.manual_backwards(loss, optimizer1)
        self.manual_backwards(loss, optimizer2)
        optimizer1.step()
        optimizer2.step()

    def training_epoch_end(self) -> None:
        """Steps the schedulers."""
        scheduler1, scheduler2 = self.lr_schedulers()
        scheduler1.step(self.trainer.callback_metrics["val_loss"])
        scheduler2.step(self.trainer.callback_metrics["val_loss"])

    def validation_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
        subset: str = "val",
    ) -> None:
        """Runs the forward pass and logs."""
        self._inference_step(batch, batch_idx, subset)

    # TODO: add test step.
    # TODO: add predict step.
