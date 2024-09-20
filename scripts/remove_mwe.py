#!/usr/bin/env python
"""Removes the "multiword" part of multi-word tokens in CoNLL-U files."""

import argparse

from udtube import data


def main(args: argparse.Namespace) -> None:
    prefix = args.source.removesuffix(".conllu")
    sink_name = prefix + "-no_mwe" + ".conllu"
    with open(sink_name, "w") as sink:
        for tokenlist in data.parse(args.source):
            # Metadata is lost during filtering, so we keep a copy.
            metadata = tokenlist.metadata
            tokenlist = tokenlist.filter(id=lambda x: not isinstance(x, tuple))
            tokenlist.metadata = metadata
            # Prevents it from adding an extra newline.
            print(tokenlist.serialize(), file=sink, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="input file")
    main(parser.parse_args())
