#!/usr/bin/env python
"""Converts Universal Dependencies features to UniMorph features."""

import argparse
import pathlib
import os
import sys

# <hack> silences an nonsensical stdout message on import.
sys.stdout = open(os.devnull, "w")
from ud_compatibility import languages, marry  # noqa: E402

sys.stdout = sys.__stdout__
# </hack>


def main(args: argparse.Namespace) -> None:
    language = languages.get_lang(args.language)
    converter = marry.FileConverter(
        pathlib.Path(args.conllu), language, clever=False
    )
    converter.convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conllu", help="CoNLL-U file to convert")
    parser.add_argument(
        "--language",
        required=True,
        help="Language (an ISO-639-1 or ISO-639-3 code or the English name)",
    )
    main(parser.parse_args())
