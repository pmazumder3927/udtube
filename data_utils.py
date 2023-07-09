def train_preprocessing(batch, tokenizer):
    """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
    sentences, *args = zip(*batch)
    # I want to load in tokens by batch to avoid using the inefficient max_length padding
    tokenized_X = tokenizer(sentences, padding='longest', return_tensors='pt')
    return tokenized_X, sentences, *args


def inference_preprocessing(sentences, tokenizer):
    """This is the case where the input is NOT a conllu file"""
    tokenized_X = tokenizer(sentences, padding='longest', return_tensors='pt')
    return tokenized_X, sentences
