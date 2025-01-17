"""Custom callbacks."""

import sys
from typing import Optional, Sequence, TextIO

from lightning.pytorch import callbacks, trainer
import torch

from . import data, models


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions in CoNLL-U format.

    If path is not specified, stdout is used. If using this in conjunction
    with > or |, add --trainer.enable_progress_bar false.

    Args:
        path: Path for the predictions file.
        model_dir: Path for checkpoints, indexes, and logs.
    """

    sink: TextIO
    mapper: data.Mapper

    def __init__(
        self,
        path: Optional[str] = None,  # If not filled in, stdout will be used.
        model_dir: str = "",  # Dummy value filled in by a link.
    ):
        super().__init__("batch")
        self.sink = open(path, "w") if path else sys.stdout
        assert model_dir, "no model_dir specified"
        self.mapper = data.Mapper.read(model_dir)

    def __del__(self):
        if self.sink is not sys.stdout:
            self.sink.close()

    # Required API.

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.UDTube,
        logits: data.Logits,
        batch_indices: Optional[Sequence[int]],
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # Batch-level argmax on the classification heads.
        upos_hat = (
            torch.argmax(logits.upos, dim=1) if logits.use_upos else None
        )
        xpos_hat = (
            torch.argmax(logits.xpos, dim=1) if logits.use_xpos else None
        )
        lemma_hat = (
            torch.argmax(logits.lemma, dim=1) if logits.use_lemma else None
        )
        feats_hat = (
            torch.argmax(logits.feats, dim=1) if logits.use_feats else None
        )
        for i, tokenlist in enumerate(batch.tokenlists):
            # Sentence-level decoding of the classification indices, followed
            # by rewriting the fields in the tokenlist.
            if upos_hat is not None:
                upos_it = self.mapper.decode_upos(upos_hat[i, :])
                for token in tokenlist:
                    if token.is_mwe:
                        continue
                    token.upos = next(upos_it)
            if xpos_hat is not None:
                xpos_it = self.mapper.decode_xpos(xpos_hat[i, :])
                for token in tokenlist:
                    if token.is_mwe:
                        continue
                    token.xpos = next(xpos_it)
            if lemma_hat is not None:
                lemma_it = self.mapper.decode_lemma(
                    tokenlist.get_tokens(), lemma_hat[i, :]
                )
                for token in tokenlist:
                    if token.is_mwe:
                        continue
                    token.lemma = next(lemma_it)
            if feats_hat is not None:
                feats_it = self.mapper.decode_feats(feats_hat[i, :])
                for token in tokenlist:
                    if token.is_mwe:
                        continue
                    token.feats = next(feats_it)
            print(tokenlist, file=self.sink)
        self.sink.flush()
