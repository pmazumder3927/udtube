"""Special symbols."""

PAD = "[PAD]"
UNK = "[UNK]"
BLANK = "_"
SPECIAL = [PAD, UNK, BLANK]

PAD_IDX = 0
UNK_IDX = 1
BLANK_IDX = 2


def isspecial(idx: int) -> bool:
    """Determines if a symbol is in the special range.

    Args:
        idx (int):

    Returns:
        bool.
    """
    return idx < len(SPECIAL)
