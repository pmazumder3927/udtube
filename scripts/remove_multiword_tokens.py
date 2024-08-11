"""Removes multiword tokenens from CoNLL-U files."""

import argparse
import glob

from typing import Any

import conllu
from udtube import defaults


def print_or_null(val: Any):
    return val if val else defaults.BLANK_TAG


def main(args: argparse.Namespace) -> None:
    for source_name in glob.iglob(f"{args.dir}/*.conllu"):
        prefix = source_name.removesuffix(".conllu")
        sink_name = prefix + "_no_mw" + ".conllu"
        # TODO: use conllu for the output too.
        with open(source_name, "r") as source, open(sink_name, "w") as sink:
            for sentence in conllu.parse_incr(
                source, field_parsers=defaults.OVERRIDDEN_FIELD_PARSERS
            ):
                print(f"# sent_id = {sentence.metadata['sent_id']}", file=sink)
                print(f"# text = {sentence.metadata['text']}", file=sink)
                for token in sentence:
                    if isinstance(token["id"], tuple):
                        continue
                    print(
                        print_or_null(token["id"]),
                        print_or_null(token["form"]),
                        print_or_null(token["lemma"]),
                        print_or_null(token["upos"]),
                        print_or_null(token["xpos"]),
                        print_or_null(str(token["feats"])),
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
    parser.add_argument("dir", help="input directory")
    main(parser.parse_args())
