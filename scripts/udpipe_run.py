import argparse
import spacy_udpipe
import conllu


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gold_file",
        help="the file which contains the correct annotations in conllu format",
    )
    parser.add_argument(
        "output_file",
        help="the file which will contain the predicted annotations in conllu format",
    )
    parser.add_argument(
        "--lang",
        help="the language (including the treebank tag, such as en-ewt). Full list of supported languages: "
             "https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/resources/languages.json"
    )
    return parser


def predict_and_write(model, gold_file, output_file):
    opened_file = open(gold_file)
    gold_data = conllu.parse_incr(opened_file)
    with open(output_file, 'w') as sink:
        for sentence in gold_data:
            print(f"# sent_id = {sentence.metadata['sent_id']}",
                  f"# text = {sentence.metadata['text']}",
                  sep="\n",
                  file=sink
                  )
            res = model(sentence.metadata["text"])
            for i, tok in enumerate(res):
                print(i + 1,  # 1-indexing, not 0-indexing
                      tok.text,
                      tok.lemma_,
                      tok.pos_,
                      tok.tag_,
                      str(tok.morph),
                      "_",
                      "_",
                      "_",  # doing nothing with this for now
                      "_",
                      sep='\t', file=sink)
            print(file=sink)
    opened_file.close()


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    spacy_udpipe.download(args.lang)
    model = spacy_udpipe.load(args.lang)
    predict_and_write(model, args.gold_file, args.output_file)