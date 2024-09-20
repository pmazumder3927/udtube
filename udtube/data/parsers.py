"""Utilities for dealing with CoNLL-U files."""

from typing import Iterator

import conllu


# Overrides the feats parser for the conllu reader so it doesn't try to
# turn them into dictionaries or None.
OVERRIDDEN_FIELD_PARSERS = {
    "feats": lambda line, i: line[i],
    "upos": lambda line, i: line[i],
    "xpos": lambda line, i: line[i],
}


def parse(path: str) -> Iterator[conllu.TokenList]:
    """Loads a CoNLL-U file and yields it sentence by sentence.

    This is useful for raw CoNLL-U input.

    Args:
        path: path to CoNLL-U file.

    Yields:
        sentences as TokenLists.
    """
    with open(path, "r") as source:
        yield from conllu.parse_incr(
            source,
            field_parsers=OVERRIDDEN_FIELD_PARSERS,
        )
