import torch
import string
import sys

from conllu_datasets import ConlluMapDataset
from lightning.pytorch.callbacks import BasePredictionWriter

from typing import Optional, Sequence, Any


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_file):
        # We only want to write at batch intervals, not epoch
        super().__init__("batch")
        self.output_file = output_file

    def _write_to_conllu(self, sentences, words, poses, xposes, lemmas, feats, replacements):
        # writing the output file
        if self.output_file == 'stdout':
            need_to_close = False
            sink = sys.stdout
        else:
            need_to_close = True
            sink = open(self.output_file, a)
        for batch_idx in range(len(words)):
            print(
                f"# sent_id = TBD",
                sep='\t', file=sink
            )
            print(
                f"# text = {sentences[batch_idx]}",
                sep='\t', file=sink
            )
            repl = replacements[batch_idx]
            for item_idx in range(len(words[batch_idx])):
                if repl:
                    # this handling is to write multword tokens
                    mws, parts = zip(*repl)
                    for i, parts in enumerate(parts):
                        p_ = parts.split()
                        if p_[0] == words[batch_idx][item_idx] and item_idx < len(words[batch_idx]) - len(p_):
                            if p_ == [w for w in words[batch_idx][item_idx:item_idx + len(p_)]]:
                                # writing multiword tok
                                print(f"{item_idx + 1}-{item_idx + len(p_) + 1}",  # 1-indexing, not 0-indexing
                                        mws[i], "_", "_", "_", "_", "_", "_", "_", "_", sep='\t', file=sink)

                space_after = "_"
                if words[batch_idx][item_idx] == ConlluMapDataset.PAD_TAG:
                    continue
                if item_idx < len(words[batch_idx]) - 1:
                    space_after = "SpaceAfter=No" if words[batch_idx][item_idx + 1] in string.punctuation else "_"
                print(item_idx + 1, # 1-indexing, not 0-indexing
                        words[batch_idx][item_idx],
                        lemmas[batch_idx][item_idx] if lemmas else "_",
                        poses[batch_idx][item_idx] if poses else "_",
                        xposes[batch_idx][item_idx] if xposes else "_",
                        feats[batch_idx][item_idx] if feats else "_",
                        "_",
                        "_",
                        "_",  # doing nothing with this for now
                        space_after,
                        sep='\t', file=sink)
            print('', file=sink)
            
            if need_to_close:
                sink.close()

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
        if batch_idx == 0:
            with open(self.output_file, 'w') as sink:
                print("# newdoc id = TBD", file=sink)
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