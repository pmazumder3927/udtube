"""simple script to write versions of conllu files without multiword tokens
Used only for the UDPipe evals"""

import argparse
import conllu
import glob

from udtube_package.__init__ import OVERRIDDEN_FIELD_PARSERS


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        help="the directory with the conllu files that need to have multiword toks removed"
    )
    return parser


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    for f in glob.glob(f"{args.dir}/*.conllu"):
        prefix = f.removesuffix(".conllu")
        new_f = prefix + "_no_mw" + ".conllu"
        data_file = open(f)
        data = conllu.parse_incr(data_file,
                                 field_parsers=OVERRIDDEN_FIELD_PARSERS)
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
                    print(tok["id"] if tok["id"] else "_",
                          tok["form"] if tok["form"] else "_",
                          tok["lemma"] if tok["lemma"] else "_",
                          tok["upos"] if tok["upos"] else "_",
                          tok["xpos"] if tok["xpos"] else "_",
                          str(tok["feats"]) if tok["feats"] else "_",
                          "_",
                          "_",
                          "_",  # doing nothing with this for now
                          "_",
                          sep='\t', file=sink)
                print(file=sink)

        data_file.close()