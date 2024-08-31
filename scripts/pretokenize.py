#!/usr/bin/env python
"""Converts data to CoNLL-U format using UDPipe models."""

import argparse

import conllu
import spacy_udpipe

BLANK = "_"


def main(args: argparse.Namespace) -> None:
    spacy_udpipe.download(args.langcode)
    model = spacy_udpipe.load(args.langcode)
    with (
        open(args.text, "r") as source,
        open(args.conllu, "w") as sink,
    ):
        document = model(source.read())
        for sentence in document.sents:
            tokenlist = conllu.TokenList(
                [
                    {
                        "id": index,
                        "form": token.text,
                        # Fills in the other fields.
                        "lemma": BLANK,
                        "upos": BLANK,
                        "xpos": BLANK,
                        "feats": BLANK,
                        "head": BLANK,
                        "deprel": BLANK,
                        "deps": BLANK,
                        "misc": BLANK,
                    }
                    for index, token in enumerate(sentence, 1)
                ],
                metadata={"text": sentence},
            )
            # Prevents it from adding an extra newline.
            print(tokenlist.serialize(), file=sink, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="path to input text file")
    parser.add_argument("conllu", help="path for output CoNLL-U file")
    parser.add_argument(
        "--langcode",
        help="the language and name of treebank (e.g., `en-ewt`); "
        "for a list of supported languages, see: "
        "https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/"
        "resources/languages.json",
    )
    main(parser.parse_args())
