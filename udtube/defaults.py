"""Defaults."""

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

# Optimizer options.
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4
BETA1 = 0.99
BETA2 = 0.999

# Scheduler options.
SCHEDULER = None
WARMUP_STEPS = 0
REDUCEONPLATEAU_FACTOR = 0.1
REDUCEONPLATEAU_PATIENCE = 10
MIN_LEARNING_RATE = 0.0
