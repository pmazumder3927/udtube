"""simple script to write versions of conllu files without multiword tokens
Used only for the UDPipe evals"""

import argparse
import conllu
import glob

from udtube_package.defaults import OVERRIDDEN_FIELD_PARSERS, BLANK_TAG


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        help="the directory with the conllu files that need to have multiword toks removed"
    )
    return parser

def print_or_null(val):
    return val if val else BLANK_TAG


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    for f in glob.iglob(f"{args.dir}/*.conllu"):
        prefix = f.removesuffix(".conllu")
        new_f = prefix + "_no_mw" + ".conllu"
        with open(f) as data_file:
            data = conllu.parse_incr(data_file, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            with open(new_f, 'w') as sink:
                for sentence in data:
                    print(f"# sent_id = {sentence.metadata['sent_id']}",
                          f"# text = {sentence.metadata['text']}",
                          sep="\n",
                          file=sink
                          )
                    for tok in sentence:
                        if isinstance(tok["id"], tuple):
                            continue
                        print(print_or_null(tok["id"]),
                              print_or_null(tok["form"]),
                              print_or_null(tok["lemma"]),
                              print_or_null(tok["upos"]),
                              print_or_null(tok["xpos"]),
                              print_or_null(str(tok["feats"])),
                              BLANK_TAG,
                              BLANK_TAG,
                              BLANK_TAG,  # doing nothing with this for now
                              BLANK_TAG,
                              sep='\t', file=sink)
                    print(file=sink)
