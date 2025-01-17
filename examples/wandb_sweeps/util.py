"""Utility functions for W&B sweep operations."""

import logging
from typing import Any, Dict


def recursive_insert(config: Dict[str, Any], key: str, value: Any) -> None:
    """Recursively inserts values into a nested dictionary.

    Args:
        config: the config dictionary.
        key: a string with the arguments separated by ".".
        value: the value to insert.
    """
    *most, last = key.split(".")
    ptr = config
    for piece in most:
        try:
            ptr = ptr[piece]
        except KeyError:
            ptr[piece] = {}
            ptr = ptr[piece]
    if last in ptr:
        logging.debug(
            "Overriding configuration argument %s with W&B sweep value: %r",
            key,
            value,
        )
    ptr[last] = value
