"""The UDTube model.

In the documentation below, N is the batch size, C is the number of classes
for a classification head, and L is the maximum length (in subwords, tokens,
or tags) of a sentence in the batch.
"""

from typing import Dict, List, Optional

import lightning
from lightning.pytorch import cli
import torch
from torch import nn
from torchmetrics import classification

from . import data, defaults, modules, special


class UDTube(lightning.LightningModule):
    """UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    tagging using a single shared, pre-trained, fine-tuned BERT-style encoder
    and sequential linear classifiers for each subtask.

    Args:
        encoder: Name of the Hugging Face model used to tokenize and encode.
        pooling_layers: Number of layers to use to compute the embedding.
        dropout: Dropout probability.
        use_upos: enables the universal POS tagging task.
        use_xpos: enables the language-specific POS tagging task.
        use_lemma: enables the lemmatization task.
        use_feats: enables the morphological feature tagging task.
    """

    encoder: modules.UDTubeEncoder
    classifier: modules.UDTubeClassifier
    loss_func: nn.CrossEntropyLoss
    # Used for validation in `fit` and testing in `test`.
    upos_accuracy: Optional[classification.MulticlassAccuracy]
    xpos_accuracy: Optional[classification.MulticlassAccuracy]
    lemma_accuracy: Optional[classification.MulticlassAccuracy]
    feats_accuracy: Optional[classification.MulticlassAccuracy]

    @staticmethod
    def _make_accuracy(num_classes: int) -> classification.MulticlassAccuracy:
        """Creates a multi-class accuracy object."""
        return classification.MulticlassAccuracy(
            num_classes, average="micro", ignore_index=special.PAD_IDX
        )

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
        encoder_optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        encoder_scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        classifier_optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        classifier_scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        # `2` is a dummy value here; it will be set by the data set object.
        upos_out_size: int = 2,
        xpos_out_size: int = 2,
        lemma_out_size: int = 2,
        feats_out_size: int = 2,
    ):
        super().__init__()
        # See what this disables here:
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#manual-optimization
        self.automatic_optimization = False
        self.encoder = modules.UDTubeEncoder(
            encoder,
            dropout,
            pooling_layers,
        )
        self.classifier = modules.UDTubeClassifier(
            self.encoder.hidden_size,
            use_upos,
            use_xpos,
            use_lemma,
            use_feats,
            upos_out_size=upos_out_size,
            xpos_out_size=xpos_out_size,
            lemma_out_size=lemma_out_size,
            feats_out_size=feats_out_size,
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)
        self.upos_accuracy = (
            self._make_accuracy(upos_out_size) if use_upos else None
        )
        self.xpos_accuracy = (
            self._make_accuracy(xpos_out_size) if use_xpos else None
        )
        self.lemma_accuracy = (
            self._make_accuracy(lemma_out_size) if use_lemma else None
        )
        self.feats_accuracy = (
            self._make_accuracy(feats_out_size) if use_feats else None
        )
        self.encoder_optimizer = encoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.classifier_optimizer = classifier_optimizer
        self.classifier_scheduler = classifier_scheduler
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
    ) -> modules.Logits:
        return self.classifier(self.encoder(batch))

    # Required API.

    def configure_optimizers(
        self,
    ) -> List[Dict]:
        """Prepare optimizers and schedulers."""
        encoder_optimizer = self.encoder_optimizer(self.encoder.parameters())
        encoder_scheduler = self.encoder_scheduler(encoder_optimizer)
        encoder_dict = {
            "optimizer": encoder_optimizer,
            "lr_scheduler": {
                "scheduler": encoder_scheduler,
                "name": "encoder_lr",
            },
        }
        classifier_optimizer = self.classifier_optimizer(
            self.classifier.parameters()
        )
        classifier_scheduler = self.classifier_scheduler(classifier_optimizer)
        classifier_dict = {
            "optimizer": classifier_optimizer,
            "lr_scheduler": {
                "scheduler": classifier_scheduler,
                "name": "classifier_lr",
            },
        }
        return [encoder_dict, classifier_dict]

    def training_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs the forward pass, logs loss, and steps the optimizers.

        Args:
            batch: a labeled batch.
            batch_idx: ignored.

        Returns:
            A tensor containing training loss.
        """
        for optimizer in self.optimizers():
            optimizer.zero_grad()
        logits = self(batch)
        losses = []
        if self.use_upos:
            losses.append(self.loss_func(logits.upos, batch.upos))
        if self.use_xpos:
            losses.append(self.loss_func(logits.xpos, batch.xpos))
        if self.use_lemma:
            losses.append(self.loss_func(logits.lemma, batch.lemma))
        if self.use_feats:
            losses.append(self.loss_func(logits.feats, batch.feats))
        loss = torch.mean(torch.stack(losses))
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.manual_backward(loss)
        for optimizer in self.optimizers():
            optimizer.step()
        return loss

    def on_train_epoch_end(self) -> None:
        """Steps the schedulers."""
        for scheduler in self.lr_schedulers():
            # Users are advised to use the subclass defined in LightningCLI,
            # which has the name of the monitored variable encoded in it.
            if isinstance(scheduler, cli.ReduceLROnPlateau):
                scheduler.step(
                    metrics=self.trainer.callback_metrics[scheduler.monitor]
                )
            else:
                scheduler.step()

    # In validation mode we compute per-batch average loss across all heads
    # and per-epoch micro-accuracy for each active head.

    # See the following for how these are called in `fit`:
    # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

    def _reset_accuracies(self) -> None:
        if self.use_upos:
            self.upos_accuracy.reset()
        if self.use_xpos:
            self.xpos_accuracy.reset()
        if self.use_lemma:
            self.lemma_accuracy.reset()
        if self.use_feats:
            self.feats_accuracy.reset()

    def on_validation_epoch_start(self) -> None:
        self._reset_accuracies()

    def validation_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
    ) -> None:
        """Runs the forward pass, updates accuracy, and logs loss.

        Args:
            batch: a labeled batch.
            batch_idx: ignored.
        """
        logits = self(batch)
        losses = []
        if self.use_upos:
            losses.append(self.loss_func(logits.upos, batch.upos))
            self.upos_accuracy.update(logits.upos, batch.upos)
        if self.use_xpos:
            losses.append(self.loss_func(logits.xpos, batch.xpos))
            self.xpos_accuracy.update(logits.xpos, batch.xpos)
        if self.use_lemma:
            losses.append(self.loss_func(logits.lemma, batch.lemma))
            self.lemma_accuracy.update(logits.lemma, batch.lemma)
        if self.use_feats:
            losses.append(self.loss_func(logits.feats, batch.feats))
            self.feats_accuracy.update(logits.feats, batch.feats)
        self.log(
            "val_loss",
            torch.mean(torch.stack(losses)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )

    def on_validation_epoch_end(self) -> None:
        """Logs accuracies for the heads."""
        if self.use_upos:
            self.log(
                "val_upos_accuracy",
                self.upos_accuracy.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        if self.use_xpos:
            self.log(
                "val_xpos_accuracy",
                self.xpos_accuracy.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        if self.use_lemma:
            self.log(
                "val_lemma_accuracy",
                self.lemma_accuracy.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        if self.use_feats:
            self.log(
                "val_feats_accuracy",
                self.feats_accuracy.compute(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    # In test mode we compute micro-accuracy for each active head.

    def on_test_step_epoch_start(self) -> None:
        self._reset_accuracies()

    def test_step(self, batch: data.ConlluBatch, batch_idx: int) -> None:
        """Runs the forward pass and updates accuracies.

        Args:
            batch: a labeled batch.
            batch_idx: ignored.
        """
        logits = self(batch)
        if self.use_upos:
            self.upos_accuracy.update(logits.upos, batch.upos)
        if self.use_xpos:
            self.xpos_accuracy.update(logits.xpos, batch.xpos)
        if self.use_lemma:
            self.lemma_accuracy.update(logits.lemma, batch.lemma)
        if self.use_feats:
            self.feats_accuracy.update(logits.feats, batch.feats)

    def on_test_epoch_end(self) -> None:
        """Logs accuracies for the heads."""
        if self.use_upos:
            self.log(
                "test_upos_accuracy",
                self.upos_accuracy.compute(),
                on_epoch=True,
                logger=True,
            )
        if self.use_xpos:
            self.log(
                "test_xpos_accuracy",
                self.xpos_accuracy.compute(),
                on_epoch=True,
                logger=True,
            )
        if self.use_lemma:
            self.log(
                "test_lemma_accuracy",
                self.lemma_accuracy.compute(),
                on_epoch=True,
                logger=True,
            )
        if self.use_feats:
            self.log(
                "test_feats_accuracy",
                self.feats_accuracy.compute(),
                on_epoch=True,
                logger=True,
            )

    # TODO: add predict step.
