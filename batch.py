import torch
from torch import nn, tensor
from typing import List, Iterable

from transformers import BatchEncoding


class TrainBatch(nn.Module):
    """The batch object used in the training step and validation step of UDTube

    The TrainBatch consists of both inputs and labels. It also handles padding inputs and tensor-ization.
    """
    def __init__(
        self,
        tokens: BatchEncoding,
        sentences: Iterable[str],
        toks: Iterable[Iterable[str]],
        pos: Iterable[tensor],
        lemma: Iterable[tensor],
        feats: Iterable[tensor],
    ):
        """Initializes the instance based on what's passed to it by the Trainer collate_fn.

       Args:
           tokens: The raw output of the BERT tokenizer
           sentences: An iterable the length of the batch that contains all the raw sentences.
           toks: An iterable the length of the batch that contains an Iterable of the token strings,
           e.g. [['hello', 'world'...]...]
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
        self.toks = toks
        self.pos = pos
        self.lemmas = lemma
        self.feats = feats

    def _pad_y(self, y, y_pad):
        """Helper function to pad and stack y_labels."""
        # all tokens sequences are same len
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
    """The batch object used in the predict step of UDTube

    The InferenceBatch consists of only tokenized inputs and sentences and does not take labels into account,
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
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)
