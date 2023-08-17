"""An evaluation script to confirm the performance of a model"""

import conllu
import argparse

from conllu_datasets import OVERRIDDEN_FIELD_PARSERS


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gold_file",
        help="the file which contains the correct annotations in conllu format",
    )
    parser.add_argument(
        "pred_file",
        help="the file which contains the predicted annotations in conllu format",
    )
    return parser


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    gold_data = conllu.parse_incr(
        args.gold_file, field_parsers=OVERRIDDEN_FIELD_PARSERS
    )
    pred_data = conllu.parse_incr(
        args.pred_file, field_parsers=OVERRIDDEN_FIELD_PARSERS
    )

    correct_pos = 0
    correct_lemma = 0
    correct_feats = 0
    total = 0

    for gold_list, pred_list in zip(gold_data, pred_data):
        assert (
            gold_list == pred_list
        ), "There's likely an issue with tokenization / file writing"
        for gold_tok, pred_tok in zip(gold_data, pred_data):
            correct_pos += gold_tok["upos"] == pred_tok["upos"]
            correct_lemma += gold_tok["lemma"] == pred_tok["lemma"]
            correct_feats += gold_tok["ufeats"] == pred_tok["ufeats"]
            total += 1

    print("POS WL ACC:", correct_pos / total)
    print("LEMMA WL ACC:", correct_lemma / total)
    print("UFEATS WL ACC:", correct_feats / total)
