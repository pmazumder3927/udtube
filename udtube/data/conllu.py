"""CoNLL-U file parser.

This is roughly compatible with the third-party package `conllu`, though it
only has features we care about."""

from __future__ import annotations

import collections
import re

from typing import Dict, Iterable, Iterator, Optional, TextIO, Tuple


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


class Error(Exception):

    pass


class ID:
    """Representation of a CoNLL-U sentence ID.

    Most of the time this is just a single integer, but MWEs use two integers
    to represent a span of tokens.

    Args:
        lower (int): the lower or sole index.
        upper (int, optional): the upper index, for MWEs.
    """

    lower: int
    upper: int

    def __init__(self, lower: int, upper: Optional[int] = None):
        self.lower = lower
        # If upper not specified, we set it to lower + 1 so this behaves
        # like an ordinary Python slice.
        self.upper = self.lower + 1 if upper is None else upper

    @classmethod
    def parse_from_string(cls, string: str) -> ID:
        if match := re.fullmatch(r"(\d+)-(\d+)", string):
            return cls(int(match.group(1)), int(match.group(2)))
        elif match := re.fullmatch(r"\d+", string):
            return cls(int(match.group()))
        else:
            raise Error(f"Unable to parse ID {str}")

    def __repr__(self) -> str:
        if self.is_mwe:
            return f"{self.__class__.__name__}({self.lower}, {self.upper})"
        return f"{self.__class__.__name__}({self.lower})"

    def __str__(self) -> str:
        if self.is_mwe:
            return f"{self.lower}-{self.upper}"
        return str(self.lower)

    def __len__(self) -> int:
        return self.upper - self.lower

    def __eq__(self, other: ID) -> bool:
        return self.lower == other.lower and self.upper == other.upper

    @property
    def is_mwe(self) -> bool:
        return len(self) > 1

    def get_slice(self) -> slice:
        return slice(self.lower, self.upper)


# TODO: One could imagine other non-string representations of the other
# columns in a CoNLL-U file but YAGNI so far.


# class Token(collections.UserDict):
#    """Token object."""


class TokenList(collections.UserList):
    """TokenList object.

    This behaves like a list of tokens (of type Dict[str, str]) with
    optional associated metadata.

    Args:
        tokens (Iterable[Dict[str, str]]): List of tokens.
        metadata (Dict[str, Optional[str]], optional): ordered dictionary of
            string/key pairs.
    """

    metadata: Dict[str, Optional[str]]

    def __init__(self, tokens: Iterable[Dict[str, str]], metadata=None):
        super().__init__(tokens)
        self.metadata = metadata if metadata is not None else {}

    def serialize(self) -> str:
        """Serializes the TokenList."""
        line_buf = []
        for key, value in self.metadata.items():
            if value:
                line_buf.append(f"# {key} = {value}")
            else:  # `newpar` etc.
                line_buf.append(f"# {key}")
        for token in self:
            col_buf = []
            for key in _fieldnames:
                col_buf.append(token.get(key, "_"))
            line_buf.append("\t".join(str(cell) for cell in col_buf))
        return "\n".join(line_buf) + "\n"


# Parsing.


def _maybe_parse_metadata(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Attempts to parse the line as metadata."""
    # The first group is the key; the optional third element is the value.
    mtch = re.fullmatch(r"#\s+(.+?)(\s+=\s+(.*))?", line)
    if mtch:
        return mtch.group(1), mtch.group(3)


def _parse_token(line: str) -> Dict[str, str]:
    """Parses the line as a token."""
    return dict(zip(_fieldnames, line.split("\t")))


def parse_from_string(buffer: str) -> TokenList:
    """Parses a CoNLL-U sentence from a string.

    Args:
        buffer: string containing a serialized sentence.

    Return:
        TokenList.
    """
    metadata = {}
    tokens = []
    for line in buffer.splitlines():
        line = line.strip()
        maybe_metadata = _maybe_parse_metadata(line)
        if maybe_metadata:
            key, value = maybe_metadata
            metadata[key] = value
        else:
            tokens.append(_parse_token(line))
    return TokenList(tokens, metadata)


def _parse_from_handle(handle: TextIO) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file from an file handle.

    This does not backtrack/rewind so it can be used with streaming inputs.

    Args:
        handle: file handle opened for reading.

    Yields:
        TokenLists.
    """
    metadata = {}
    tokens = []
    for line in handle:
        line = line.strip()
        if not line:
            if tokens or metadata:
                yield TokenList(tokens.copy(), metadata.copy())
                metadata.clear()
                tokens.clear()
            continue
        maybe_metadata = _maybe_parse_metadata(line)
        if maybe_metadata:
            key, value = maybe_metadata
            metadata[key] = value
        else:
            tokens.append(_parse_token(line))
    if tokens or metadata:
        # No need to take a copy for the last one.
        yield TokenList(tokens, metadata)


def parse_from_path(path: str) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file from an file path.

    This does not backtrack/rewind so it can be used with streaming inputs.

    Args:
        path: path to input CoNLL-U file.

    Yields:
        TokenLists.
    """
    with open(path, "r") as source:
        yield from _parse_from_handle(source)
