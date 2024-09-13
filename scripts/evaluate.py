#!/usr/bin/env python
"""UDTube evaluation."""

import argparse
import logging

from udtube import data


def main(args: argparse.Namespace) -> None:
    correct_upos = 0
    correct_xpos = 0
    correct_lemma = 0
    correct_feats = 0
    total = 0
    for gold_list, hypo_list in zip(
        data.parse(args.gold), data.parse(args.hypo)
    ):
        gold_idx = 0
        hypo_idx = 0
        for _ in zip(gold_list, hypo_list):
            try:
                gold_tok = gold_list[gold_idx]
                hypo_tok = hypo_list[hypo_idx]
            except IndexError:
                logging.warning(
                    "Sequence mismatch: %r and %r", gold_list, hypo_list
                )
                total += 1
                continue
            # Attempts to fix alignment errors within +/- 1 non-UNK
            # word.
            # TODO: is this still necessary with the current MWE handling?
            if (
                gold_tok["form"] != hypo_tok["form"]
                and "[UNK]" != hypo_tok["form"]
            ):
                if len(hypo_list) > hypo_idx + 1:
                    if gold_tok["form"] == hypo_list[hypo_idx + 1]["form"]:
                        hypo_idx += 1
                        hypo_tok = hypo_list[hypo_idx]
                if len(gold_list) > gold_idx + 1:
                    if gold_list[gold_idx + 1] == hypo_tok["form"]:
                        gold_idx += 1
                        gold_tok = gold_list[gold_idx]
            # Does not check multiword tokens, since they are unannotated.
            if isinstance(gold_tok["id"], tuple):
                if isinstance(hypo_tok["id"], tuple):
                    continue
            correct_upos += gold_tok["upos"] == hypo_tok["upos"]
            correct_xpos += gold_tok["xpos"] == hypo_tok["xpos"]
            # We do this case-insensitively.
            correct_lemma += (
                gold_tok["lemma"].casefold() == hypo_tok["lemma"].casefold()
            )
            correct_feats += gold_tok["feats"] == hypo_tok["feats"]
            total += 1
            gold_idx += 1
            hypo_idx += 1
    # Expresses this as a percentage as per the community preference.
    logging.info("UPOS accuracy:\t%2.2f", 100 * correct_upos / total)
    logging.info("XPOS accuracy:\t%2.2f", 100 * correct_xpos / total)
    logging.info("LEMMA accuracy:\t%2.2f", 100 * correct_lemma / total)
    logging.info("FEATS accuracy:\t%2.2f", 100 * correct_feats / total)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", help="path to gold .conllu file")
    parser.add_argument("hypo", help="path to hypo .conllu file")
    main(parser.parse_args())
