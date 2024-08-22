"""Additional callbacks."""

import string
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch

from . import data, defaults


# TODO: update this with the new cool shit.


class CustomWriter(pl.callbacks.BasePredictionWriter):
    """Writes results after each batch.

    Following our a generally incremental philosophy, this modifies the default
    on_test_batch_end behavior so that after every batch results are written
    to a file (or stdout).
    """

    # TODO(#18): address complexity here.

    def __init__(self, output_file):
        # We only want to write at batch intervals, not epoch
        super().__init__("batch")
        self.output_file = output_file

    def _write_to_conllu(
        self,
        sentences: Iterable[str],
        words: Iterable[str],
        poses: Iterable[List[int]],
        xposes: Iterable[List[int]],
        lemmas: Iterable[List[int]],
        feats: Iterable[List[int]],
        replacements: Iterable[Tuple[str, str]],
    ):
        # writing the output file
        if self.output_file == "stdout":
            need_to_close = False
            sink = sys.stdout
        else:
            need_to_close = True
            sink = open(self.output_file, "a")
        for batch_idx in range(len(words)):
            print("# sent_id = TBD", sep="\t", file=sink)
            print(f"# text = {sentences[batch_idx]}", sep="\t", file=sink)
            repl = replacements[batch_idx]
            for item_idx in range(len(words[batch_idx])):
                if repl:
                    # this handling is to write multword tokens
                    mws, parts = zip(*repl)
                    for i, parts in enumerate(parts):
                        p_ = parts.split()
                        if p_[0] == words[batch_idx][
                            item_idx
                        ] and item_idx < len(words[batch_idx]) - len(p_):
                            if p_ == [
                                w
                                for w in words[batch_idx][
                                    item_idx : item_idx + len(p_)
                                ]
                            ]:
                                # writing multiword tok
                                # 1-indexing, not 0-indexing
                                print(
                                    f"{item_idx + 1}-{item_idx + len(p_) + 1}",
                                    mws[i],
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    sep="\t",
                                    file=sink,
                                )
                space_after = "_"
                if words[batch_idx][item_idx] == defaults.PAD_TAG:
                    continue
                # TODO(#19): this is probably broken outside of English.
                if item_idx < len(words[batch_idx]) - 1:
                    space_after = (
                        "SpaceAfter=No"
                        if words[batch_idx][item_idx + 1] in string.punctuation
                        else "_"
                    )
                print(
                    item_idx + 1,  # 1-indexing, not 0-indexing
                    words[batch_idx][item_idx],
                    lemmas[batch_idx][item_idx] if lemmas else "_",
                    poses[batch_idx][item_idx] if poses else "_",
                    xposes[batch_idx][item_idx] if xposes else "_",
                    feats[batch_idx][item_idx] if feats else "_",
                    "_",
                    "_",
                    "_",  # Doing nothing with this for now
                    space_after,
                    sep="\t",
                    file=sink,
                )
            print(file=sink)
            if need_to_close:
                sink.close()

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Tuple[Iterable[str], list[list[str]], torch.Tensor],
        batch_indices: Optional[Sequence[int]],
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._write_to_conllu(*prediction)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tuple[Iterable[str], list[list[str]], torch.Tensor],
        batch: batches.Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(
            trainer,
            pl_module,
            outputs,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx,
        )
