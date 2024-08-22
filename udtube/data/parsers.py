"""Utilities for dealing with CoNLL-U files.

This eliminates most of the need to work with finicky third-party conllu module
directly.
"""

import dataclasses
from typing import Iterator, List, Optional, Tuple

import conllu

from .. import defaults
from . import edit_scripts

# Overrides the feats parser for the conllu reader so it doesn't try to
# turn them into dictionaries.
OVERRIDDEN_FIELD_PARSERS = {
    "feats": lambda line, i: line[i],
}


SampleType = Tuple[
    str,
    List[str],
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[str]],
]


def tokenlist_iterator(path: str) -> Iterator[conllu.TokenList]:
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


@dataclasses.dataclass
class ConlluParser:
    """Streams relevant data from a CoNLL-U file."""

    use_upos: bool
    use_xpos: bool
    use_lemma: bool
    use_feats: bool
    script: edit_scripts.EditScript

    def __init__(
        self,
        use_upos=defaults.USE_UPOS,
        use_xpos=defaults.USE_XPOS,
        use_lemma=defaults.USE_LEMMA,
        use_feats=defaults.USE_FEATS,
        reverse_edits=defaults.REVERSE_EDITS,
    ):
        self.use_upos = use_upos
        self.use_xpos = use_xpos
        self.use_lemma = use_lemma
        self.use_feats = use_feats
        self.script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )

    # TODO: look into whether casefolding is the thing to do here. At least
    # some UD corpora use titlecase to indicate things like proper noun-hood,
    # but we casefold before any lemma processing. Evaluation scripts will
    # need to take this into account to be sensible.

    def lemma_rule(self, form: str, lemma: str) -> str:
        """Computes the lemma tag."""
        return str(self.script(form.casefold(), lemma.casefold()))

    def lemmatize(self, form: str, tag: str) -> str:
        """Applies the lemma tag to a form."""
        rule = self.script.fromtag(tag)
        return rule.apply(form.casefold())

    def samples(self, path: str) -> Iterator[SampleType]:
        for tokenlist in tokenlist_iterator(path):
            form = [token["form"] for token in tokenlist]
            upos = None
            if self.use_upos:
                upos = [token["upos"] for token in tokenlist]
            xpos = None
            if self.use_xpos:
                xpos = [token["xpos"] for token in tokenlist]
            lemma = None
            if self.use_lemma:
                lemma = []
                for token in tokenlist:
                    lemma.append(
                        self.lemma_rule(
                            token["form"], token["lemma"].casefold()
                        )
                    )
            feats = None
            if self.use_feats:
                feats = [token["feats"] for token in tokenlist]
            yield tokenlist.metadata["text"], form, upos, xpos, lemma, feats
