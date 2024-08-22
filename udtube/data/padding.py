"""Padding helpers."""

from typing import List

import torch
from torch import nn


def pad_tensors(
    tensorlist: List[torch.Tensor],
    pad_idx: int = 0,
) -> torch.Tensor:
    """Pads and stacks a list of tensors.

    Args:
        tensorlist: a list of tensors to be padded.
        pad_idx: padding index to use; defaults to 0.

    Returns:
        The padded and stacked tensor.
    """
    pad_max = max(len(tensor) for tensor in tensorlist)
    return torch.stack(
        [
            nn.functional.pad(
                tensor, (0, pad_max - len(tensor)), "constant", pad_idx
            )
            for tensor in tensorlist
        ]
    )
