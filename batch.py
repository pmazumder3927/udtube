import torch
from torch import nn, tensor


class TrainBatch(nn.Module):
    def __init__(self, tokens, sentences, toks, pos, lemma, ufeats):
        self.tokens = tokens
        self.sentences = sentences
        self.toks = toks
        self.pos = pos
        self.lemmas = lemma
        self.ufeats = ufeats

    def _pad_y(self, y, y_pad):
        longest_sequence = len(
            self.tokens[0]
        )  # all token sequences are same len
        padded_y = []
        for y_i in y:
            l_padding = tensor([y_pad])  # for [CLS] token
            r_padding = tensor([y_pad] * (longest_sequence - len(y_i) - 1))
            padded_y.append(torch.cat((l_padding, y_i, r_padding)))
        return torch.stack(padded_y)

    def __len__(self):
        return len(self.ids)


class InferenceBatch(nn.Module):
    def __init__(self, tokens, sentences):
        self.tokens = tokens
        self.sentences = sentences

    def __len__(self):
        return len(self.ids)
