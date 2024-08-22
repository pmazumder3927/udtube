"""Padding helpers."""

from typing import List

import torch
from torch import nn


def pad_tensors(
    tensorlist: List[torch.Tensor], pad_idx: int = 0,
    pad_max: Optional[int] = None,
) -> torch.Tensor:
    """Pads and stacks a list of tensors.

    Args:
        tensorlist: a list of tensors to be padded.
        pad_idx: padding index to use; defaults to 0.
        pad_max: maximum length; computed from the tensor list if not
            provided.

    Returns:
        The padded and stacked tensor.
    """
    if pad_max is None:
        pad_max = max(len(tensor) for tensor in tensorlist)
    return torch.stack(
        [
            nn.functional.pad(
                tensor, (0, pad_max - len(tensor)), "constant", pad_idx
            )
            for tensor in tensorlist
        ]
    )
