import torch
from torch import nn, tensor
from typing import List, Iterable

from transformers import BatchEncoding


class ConlluBatch(nn.Module):
    """The batch object used in the training step and validation step of UDTube

    The ConlluBatch consists of both inputs and labels. It also handles padding inputs and tensor-ization.
    """
    def __init__(
        self,
        tokens: BatchEncoding,
        sentences: Iterable[str],
        pos: Iterable[List[int]],
        lemma: Iterable[List[int]],
        feats: Iterable[List[int]],
        head: Iterable[List[int]],
        deprel: Iterable[List[int]]
    ):
        """Initializes the instance based on what's passed to it by the Trainer collate_fn.

       Args:
           tokens: The raw output of the BERT tokenizer
           sentences: An iterable the length of the batch that contains all the raw sentences.
           pos: An iterable the length of the batch that contains a tensor of POS labels.
           Each tensor is as long as the respective entry in toks.
           lemma: An iterable the length of the batch that contains a tensor of lemma labels.
           Each tensor is as long as the respective entry in toks.
           feats: An iterable the length of the batch that contains a tensor of feats labels.
           Each tensor is as long as the respective entry in toks.
       """
        super().__init__()
        self.tokens = tokens
        self.sentences = sentences
        self.pos = pos
        self.lemmas = lemma
        self.feats = feats
        self.heads = head
        self.deprels = deprel

    def __len__(self):
        return len(self.sentences)


class TextBatch(nn.Module):
    """The batch object used in the predict step of UDTube

    The TextBatch consists of only tokenized inputs and sentences and does not take labels into account,
    distinct from TrainBatch
    """
    def __init__(self, tokens: BatchEncoding, sentences: List[str]):
        """Initializes the instance based on what's passed to it by the Trainer collate_fn.

       Args:
           tokens: The raw output of the BERT tokenizer
           sentences: An iterable the length of the batch that contains all the raw sentences.
       """
        super().__init__()
        self.tokens = tokens
        self.sentences = [s.strip("\n") for s in sentences]

    def __len__(self):
        return len(self.sentences)

