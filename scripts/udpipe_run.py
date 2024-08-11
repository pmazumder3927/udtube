"""Spacy-udpipe runner."""

import argparse

import conllu
import spacy_udpipe

from udtube import defaults


def main(args: argparse.Namespace) -> None:
    spacy_udpipe.download(args.lang)
    model = spacy_udpipe.load(args.lang)
    with (
        open(args.gold_file, "r") as source,
        open(args.output_file, "w") as sink,
    ):
        for sentence in conllu.parse_incr(source):
            print(f"# sent_id = {sentence.metadata['sent_id']}", file=sink)
            print(f"# text = {sentence.metadata['text']}", file=sink)
            result = model(sentence.metadata["text"])
            # TODO: use conllu for the output too.
            for i, token in enumerate(result):
                print(
                    i + 1,  # 1-indexing.
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    str(token.morph),
                    defaults.BLANK_TAG,
                    defaults.BLANK_TAG,
                    defaults.BLANK_TAG,
                    defaults.BLANK_TAG,
                    sep="\t",
                    file=sink,
                )
            print(file=sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold_file", help="path input CoNLL-U file")
    parser.add_argument("output_file", help="path for output CoNLL-U file")
    parser.add_argument(
        "--langcode",
        help="the language and name of treebank (e.g., `en-ewt`); "
        "for a list of supported languages, see: "
        "https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/"
        "resources/languages.json",
    )
    main(parser.parse_args())
