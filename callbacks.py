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
                    if words[batch_idx][item_idx] == ConlluMapDataset.PAD_TAG:
                        continue
                    if item_idx < len(words[batch_idx]) - 1:
                        space_after = "SpaceAfter=No" if words[batch_idx][item_idx + 1] in string.punctuation else ""
                    print(item_idx,
                          words[batch_idx][item_idx],
                          lemmas[batch_idx][item_idx],
                          poses[batch_idx][item_idx],
                          xposes[batch_idx][item_idx],
                          feats[batch_idx][item_idx],
                          s_arcs[batch_idx][item_idx],
                          s_rels[batch_idx][item_idx],
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
        sentences, words, poses, xposes, lemmas, feats = prediction
        self._write_to_conllu(sentences, words, poses, xposes, lemmas, feats)