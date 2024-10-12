"""CoNLL-U file parser

This is roughly compatible with the third-party package `conllu`, though it
only has features we care about."""

import re

from typing import Dict, Iterable, Iterator, List, TextIO


# From: https://universaldependencies.org/format.html.
_fieldnames = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
]


class TokenList:
    """TokenList object.

    Args:
        tokens: List of tokens.
        metadata: ordered dictionary of string/key pairs.
    """

    tokens: List[Dict[str, str]]
    metadata: Dict[str, str]

    def __init__(self, tokens: Iterable[Dict[str, str]], metadata=None):
        self.tokens = list(tokens)
        self.metadata = metadata if metadata is not None else {}

    def serialize(self) -> str:
        """Serializes the TokenList."""
        line_buf = []
        for key, value in self.metadata.items():
            line_buf.append(f"# {key}: {value}")
        for token in self.tokens:
            col_buf = []
            for key in _fieldnames:
                col_buf.append(token.get(key, "_"))
            line_buf.append("\t".join(col_buf))
        return "\n".join(line_buf) + "\n"

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.tokens)

    def __setitem__(self, index: int, value: Dict[str, str]) -> None:
        self.tokens[index] = value

    def append(self, token: Dict[str, str]) -> None:
        self.tokens.append(token)


# Parsing.


def parse_from_string(buffer: str) -> TokenList:
    """Parses a CoNLL-U sentence from a string.

    Args:
        buffer: string containing a serialized sentence.

    Return:
        TokenList.
    """
    # Unfortunately it wasn't obvious how to do this without
    # repeating some of the logic in the following function.
    metadata = {}
    tokens = []
    for line in buffer.splitlines():
        line = line.strip()
        mtch = re.fullmatch(r"#\s+(.+?)\s+=\s+(.*)", line)
        if mtch:
            metadata[mtch.group(1)] = mtch.group(2)
        if not mtch:
            tokens.append(dict(zip(_fieldnames, line.split("\t"))))
    return TokenList(tokens, metadata)


def _parse_from_handle(handle: TextIO) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file from an file handle.

    This does not backtrack/rewind so it can be used with streaming inputs.

    Args:
        handle: file handle opened for reading.

    Yields:
        TokenLists.
    """
    metadata_buf = {}
    token_buf = []
    for line in handle:
        line = line.strip()
        if not line:
            if token_buf or metadata_buf:
                yield TokenList(token_buf.copy(), metadata_buf.copy())
                metadata_buf.clear()
                token_buf.clear()
            continue
        mtch = re.fullmatch(r"#\s+(.+?)\s+=\s+(.*)", line)
        if mtch:
            metadata_buf[mtch.group(1)] = mtch.group(2)
        if not mtch:
            token_buf.append(dict(zip(_fieldnames, line.split("\t"))))
    if token_buf or metadata_buf:
        yield TokenList(token_buf, metadata_buf)


def parse_from_path(path: str) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file.

    Args:
        path: path to input CoNLL-U file.

    Yields:
        TokenLists.
    """
    with open(path, "r") as source:
        yield from _parse_from_handle(source)
