"""Custom callbacks."""

import conllu
from lightning.pytorch import callbacks, trainer

from . import data, special


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Hardcodes some of the options for ModelCheckpoint."""

    def __init__(self, dirname: str, save_top_k: int):
        super().__init__(
            dirname,
            filename="{epoch:03d}-{val_loss:.2f}",
            save_top_k=save_top_k,
            mode="min",
            monitor="val_loss",
        )


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions in CoNLL-U format."""

    def __init__(self, predictions: str):
        super().__init__("batch")
        self.sink = open(predictions, "w")

    def __del__(self):
        self.sink.close()

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: lightning.LightningModule,
        logits: schedule.Logits,
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        for sentence_logits, sentence in zip(logits, batch.texts):
            tokenlist = conllu.TokenList(
                [
                    {
                        "id": index,
                        "form": special.BLANK,  # FIXME
                        "lemma": special.BLANK,  # FIXME
                        "upos": special.BLANK,  # FIXME
                        "xpos": special.BLANK,  # FIXME
                        "head": special.BLANK,
                        "deprel": special.BLANK,
                        "deps": special.BLANK,
                        "misc": special.BLANK,
                    }
                    for index, token in enumerate(sentence_logits, 1)
                ],
                metadata={"text": sentence},
            )
            # Prevents it from adding an extra newline.
            print(tokenlist.serialize(), file=self.sink, end="")
        self.sink.flush()
