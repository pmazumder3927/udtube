"""Defaults."""

import conllu

# Overriding the feats parser for the conllu reader.
OVERRIDDEN_FIELD_PARSERS = {
    "feats": lambda line, i: conllu.parser.parse_nullable_value(line[i])
}
UNK_TAG = "[UNKNOWN]"
PAD_TAG = "[PAD]"
BLANK_TAG = "_"
SPECIAL_TOKENS = [UNK_TAG, PAD_TAG, BLANK_TAG]

LANGUAGE = "en"

BATCH_SIZE = 32

REVERSE_EDITS = True
LANG_WITH_SPACE = True

ENCODER = "bert-base-multilingual-cased"
ENCODER_DROPOUT = 0.5
ENCODER_LEARNING_RATE = 5e-3
POOLING_LAYERS = 4

CLASSIFIER_DROPOUT = 0.3
CLASSIFIER_LEARNING_RATE = 2e-5

WARMUP_STEPS = 500

USE_POS = True
USE_XPOS = True
USE_LEMMA = True
USE_FEATS = True
