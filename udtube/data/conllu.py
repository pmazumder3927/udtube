"""CoNLL-U file parser

This is roughly compatible with the third-party package `conllu`, though it
only has features we care about, and supports streaming inputs."""

import re

from typing import Dict, Iterable, Iterator, List


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

    tokens: List[Dict]
    metadata: Dict[str, str]

    def __init__(self, tokens: Iterable[Dict], metadata=None):
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

    def __getitem__(self, index: int) -> Dict:
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.tokens)

    def __setitem__(self, index: int, value: Dict) -> None:
        self.tokens[index] = value

    def append(self, token: Dict) -> None:
        self.tokens.append(token)


def parse(path: str) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file.

    Args:
        path: path to input CoNLL-U file.

    Yields:
        TokenLists.
    """
    metadata_buf = {}
    token_buf = []
    with open(path, "r") as source:
        for line in source:
            line = line.strip()
            if not line:
                if token_buf or metadata_buf:
                    yield TokenList(token_buf.copy(), metadata_buf.copy())
                    metadata_buf.clear()
                    token_buf.clear()
                continue
            match = re.fullmatch(r"#\s+(.+?)\s+=\s+(.*)", line)
            if match:
                metadata_buf[match.group(1)] = match.group(2)
            if not match:
                token_buf.append(dict(zip(_fieldnames, line.split("\t"))))
        if token_buf or metadata_buf:
            yield TokenList(token_buf, metadata_buf)
