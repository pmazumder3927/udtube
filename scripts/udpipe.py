#!/usr/bin/env python
"""Runs UDPipe over CoNLL-U data."""

import argparse

import conllu
import spacy_udpipe


BLANK = "_"


def main(args: argparse.Namespace) -> None:
    spacy_udpipe.download(args.langcode)
    model = spacy_udpipe.load(args.langcode)
    with open(args.input, "r") as source, open(args.output, "w") as sink:
        for sentence in conllu.parse_incr(source):
            # The model insists on retokenizing the data but this is harmless.
            result = model(sentence.metadata["text"])
            tokenlist = conllu.TokenList(
                [
                    {
                        "id": index,
                        "form": token.text,
                        "lemma": token.lemma_,
                        "upos": token.pos_,
                        "xpos": token.tag_,
                        "feats": str(token.morph),
                        "head": BLANK,
                        "deprel": BLANK,
                        "deps": BLANK,
                        "misc": BLANK,
                    }
                    for index, token in enumerate(result, 1)
                ],
                metadata=sentence.metadata,
            )
            # Prevents it from adding an extra newline.
            print(tokenlist.serialize(), file=sink, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to input CoNLL-U file")
    parser.add_argument("output", help="path for output CoNLL-U file")
    parser.add_argument(
        "--langcode",
        required=True,
        help="the language and name of treebank (e.g., `en-ewt`); "
        "for a list of supported languages, see: "
        "https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/"
        "resources/languages.json",
    )
    main(parser.parse_args())
