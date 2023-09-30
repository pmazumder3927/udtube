import torch
import string

from conllu_datasets import ConlluMapDataset
from lightning.pytorch.callbacks import BasePredictionWriter

from typing import Optional, Sequence, Any


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_file):
        super().__init__("batch") # We only want to write at batch intervals, not epoch
        self.output_file = output_file

    def _write_to_conllu(self, sentences, words, poses, xposes, lemmas, feats, s_arcs, s_rels):
        # writing the output file
        with open(self.output_file, 'a') as sink:
            for batch_idx in range(len(words)):
                print(
                    f"# text = {sentences[batch_idx]}",
                    sep='\t', file=sink
                )
                for item_idx in range(len(words[batch_idx])):
                    space_after = "_"
                    if words[batch_idx][item_idx] == ConlluMapDataset.PAD_TAG:
                        continue
                    if item_idx < len(words[batch_idx]) - 1:
                        space_after = "SpaceAfter=No" if words[batch_idx][item_idx + 1] in string.punctuation else "_"
                    print(item_idx + 1, # 1-indexing, not 0-indexing
                          words[batch_idx][item_idx],
                          lemmas[batch_idx][item_idx],
                          poses[batch_idx][item_idx],
                          xposes[batch_idx][item_idx],
                          feats[batch_idx][item_idx],
                          s_arcs[batch_idx][item_idx],
                          s_rels[batch_idx][item_idx],
                          "_",  # doing nothing with this for now
                          space_after,
                          sep='\t', file=sink)
                print('', file=sink)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._write_to_conllu(*prediction)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)