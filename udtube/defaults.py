"""Defaults."""

from torch import optim

BATCH_SIZE = 32

REVERSE_EDITS = True

LANGUAGE = "en"  # Sigh.

# Encoder options.
ENCODER = "bert-base-multilingual-cased"
DROPOUT = 0.5
POOLING_LAYERS = 4

# Classifier options.
USE_UPOS = True
USE_XPOS = True
USE_LEMMA = True
USE_FEATS = True

# Optimization options.
OPTIMIZER = optim.Adam
SCHEDULER = optim.lr_scheduler.ConstantLR
