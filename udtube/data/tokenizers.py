"""Loads tokenizers."""

import transformers


def load(model_name: str) -> transformers.AutoTokenizer:
    """Loads the tokenizer.

    Args:
        model_name: the Hugging Face model name.

    Returns:
        A Hugging Face tokenizer.
    """
