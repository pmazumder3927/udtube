import torch
from torch import nn, tensor
from typing import List, Iterable

from transformers import BatchEncoding


class TrainBatch(nn.Module):
    """The batch object used in the training step and validation step of UDTube"""

    def __init__(
        self,
        tokens: BatchEncoding,
        sentences: Iterable[str],
        toks: Iterable[Iterable[str]],
        pos: Iterable[tensor],
        lemma: Iterable[tensor],
        ufeats: Iterable[tensor],
    ):
        super().__init__()
        self.tokens = tokens
        self.sentences = sentences
        self.toks = toks
        self.pos = pos
        self.lemmas = lemma
        self.ufeats = ufeats

    def _pad_y(self, y, y_pad):
        # all token sequences are same len
        longest_sequence = len(self.tokens[0])
        padded_y = []
        for y_i in y:
            l_padding = tensor([y_pad])  # for [CLS] token
            r_padding = tensor([y_pad] * (longest_sequence - len(y_i) - 1))
            padded_y.append(torch.cat((l_padding, y_i, r_padding)))
        return torch.stack(padded_y)

    def __len__(self):
        return len(self.sentences)


class InferenceBatch(nn.Module):
    """The batch object used in the forward pass of UDTube. It does not contain y labels"""

    def __init__(self, tokens: BatchEncoding, sentences: List[str]):
        super().__init__()
        self.tokens = tokens
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)
