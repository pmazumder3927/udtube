from torch import nn


class TrainBatch(nn.Module):
    def __init__(self, ids, word_mappings, attention_masks, sentences, toks, pos, lemma, ufeats):
        self.ids = ids
        self.word_mappings = word_mappings
        self.attention_masks = attention_masks
        self.sentences = sentences
        self.toks = toks
        self.pos = pos
        self.lemmas = lemma
        self.ufeats = ufeats

    def __len__(self):
        return len(self.ids)


class InferenceBatch(nn.Module):
    def __init__(self, ids, word_mappings, attention_masks, sentences, toks):
        self.ids = ids
        self.word_mappings = word_mappings
        self.attention_masks = attention_masks
        self.sentences = sentences
        self.toks = toks

    def __len__(self):
        return len(self.ids)
