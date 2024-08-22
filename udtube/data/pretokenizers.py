"""Loads pretokenizers."""

import spacy
import spacy_udpipe


class Error(Exception):
    pass


def load(language: str) -> spacy.language.Language:
    """Loads the pretokenizer.

    Args:
        language: ISO-639-1 language code.

    Returns:
        A Spacy model with a pretokenizer.

    Raises:
        Error: language not available.
    """
    # Annoyingly load failure raises an AssertionError. We hack around it.
    try:
        return spacy_udpipe.load(language)
    except AssertionError:
        try:
            spacy_udpipe.download(language)
            return spacy_udpipe.load(language)
        except AssertionError as error:
            raise Error(error)
