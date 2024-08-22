"""Special symbols."""

PAD = "[PAD]"
UNK = "[UNK]"
BLANK = "_"
SPECIALS = [PAD, UNK, BLANK]

# This is guaranteed to be true in the indexing so long as it is the first in
# SPECIALS above.
PAD_IDX = 0
