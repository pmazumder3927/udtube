from transformers import AutoTokenizer
from batch import TrainBatch, InferenceBatch


def train_preprocessing(batch, tokenizer: AutoTokenizer):
    """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
    sentences, *args = zip(*batch)
    # I want to load in tokens by batch to avoid using the inefficient max_length padding
    tokenized_x = tokenizer(sentences, padding="longest", return_tensors="pt")
    return TrainBatch(tokenized_x, sentences, *args)


def inference_preprocessing(sentences, tokenizer: AutoTokenizer):
    """This is the case where the input is NOT a conllu file"""
    tokenized_x = tokenizer(sentences, padding="longest", return_tensors="pt")
    return InferenceBatch(tokenized_x, sentences)
