"""Custom callbacks."""

from typing import Optional, Sequence, TextIO

from lightning.pytorch import callbacks, trainer
import torch

from . import data, models


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions in CoNLL-U format.

    Args:
        predictions: path for the predictions file.
    """

    sink: TextIO
    mapper: data.Mapper

    def __init__(self, predictions: str, model_dir: str):
        super().__init__("batch")
        self.sink = open(predictions, "w")
        self.mapper = data.Mapper.read(model_dir)

    def __del__(self):
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
        y_upos_hat = (
            torch.argmax(logits.upos, dim=1) if logits.use_upos else None
        )
        y_xpos_hat = (
            torch.argmax(logits.xpos, dim=1) if logits.use_xpos else None
        )
        y_lemma_hat = (
            torch.argmax(logits.lemma, dim=1) if logits.use_lemma else None
        )
        y_feats_hat = (
            torch.argmax(logits.feats, dim=1) if logits.use_feats else None
        )
        for i, tokenlist in enumerate(batch.tokenlists):
            # Edits the relevant fields in each tokenlist.
            if y_upos_hat is not None:
                upos_hat = self.mapper.decode_upos(y_upos_hat[i, :])
                for j, upos in enumerate(upos_hat[: len(tokenlist)]):
                    tokenlist[j]["upos"] = upos
            if y_xpos_hat is not None:
                xpos_hat = self.mapper.decode_xpos(y_xpos_hat[i, :])
                for j, xpos in enumerate(xpos_hat[: len(tokenlist)]):
                    tokenlist[j]["xpos"] = xpos
            if y_lemma_hat is not None:
                lemma_hat = self.mapper.decode_lemma(
                    [token["form"] for token in tokenlist],
                    y_lemma_hat[i, :],
                )
                # `zip` should have already cut off the padding, so
                # subscripting is not necessary here.
                assert len(lemma_hat) == len(tokenlist)
                for j, lemma in enumerate(lemma_hat):
                    tokenlist[j]["lemma"] = lemma
            if y_feats_hat is not None:
                feats_hat = self.mapper.decode_feats(y_feats_hat[i, :])
                for j, feats in enumerate(feats_hat[: len(tokenlist)]):
                    tokenlist[j]["feats"] = feats
            # Prevents it from adding an extra newline.
            print(tokenlist.serialize(), file=self.sink, end="")
        self.sink.flush()
