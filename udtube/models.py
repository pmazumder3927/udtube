"""The UDTube model.

In the documentation below, N is the batch size, C is the number of classes
for a classification head, and L is the maximum length (in subwords, tokens,
or tags) of a sentence in the batch.
"""

import inspect
from typing import Dict, List

import lightning
from lightning.pytorch import cli
import torch
from torch import nn, optim
from torchmetrics.functional import classification

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
        # Other stuff.
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)
        self.save_hyperparameters()
        self.encoder_optimizer = encoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.classifier_optimizer = classifier_optimizer
        self.classifier_scheduler = classifier_scheduler

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
                "interval": "epoch",
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
                "interval": "epoch",
                "name": "classifier_lr",
            },
        }
        return [encoder_dict, classifier_dict]

    @staticmethod
    def _accuracy(logits: torch.Tensor, golds: torch.Tensor) -> float:
        """Computes multi-class micro-accuracy for a head.

        Args:
            logits: logits, tensor of shape N x C x L.
            golds: gold data, tensor of shape N x L.

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

    def _log_accuracy(
        self,
        logits: torch.Tensor,
        golds: torch.Tensor,
        task_name: str,
        subset: str = "train",
    ) -> None:
        """Computes and logs accuracy for a head.
        
        Args:
            logits: logits, tensor of shape N x C x L.
            golds: gold data, tensor of shape N x L.
            subset: one of "train", "val", "predict" or "test".
        """
        self.log(
            f"{subset}_{task_name}_accuracy",
            self._accuracy(logits, golds),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(logits),
        )

    def training_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
        subset: str = "train",
    ) -> torch.Tensor:
        """Runs the forward pass and logs loss.

        Args:
            batch: a labeled batch.
            batch_idx: ignored.
            subset: "train"

        Returns:
            A tensor containing training loss.
        """
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
            f"{subset}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    def validation_step(
        self,
        batch: data.ConlluBatch,
        batch_idx: int,
        subset: str = "val",
    ) -> torch.Tensor:
        """Runs the forward pass and logs loss and accuracy.

        Args:
            batch: a labeled batch.
            batch_idx: ignored.
            subset: "val"

        Returns:
            A tensor containing validation loss.
        """
        logits = self(batch)
        losses = []
        if self.use_upos:
            losses.append(self.loss_func(logits.upos, batch.upos))
            self._log_accuracy(
                logits.upos,
                batch.upos,
                task_name="upos",
                subset=subset,
            )
        if self.use_xpos:
            losses.append(self.loss_func(logits.xpos, batch.xpos))
            self._log_accuracy(
                logits.xpos,
                batch.xpos,
                task_name="xpos",
                subset=subset,
            )
        if self.use_lemma:
            losses.append(self.loss_func(logits.lemma, batch.lemma))
            self._log_accuracy(
                logits.lemma,
                batch.lemma,
                task_name="lemma",
                subset=subset,
            )
        if self.use_feats:
            losses.append(self.loss_func(logits.feats, batch.feats))
            self._log_accuracy(
                logits.feats,
                batch.feats,
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

    # TODO: add test step.
    # TODO: add predict step.
