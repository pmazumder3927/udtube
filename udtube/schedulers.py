"""Custom schedulers."""

import numpy

from torch import optim


class WarmupInverseSquareRoot(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup steps, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.

    Args:
        optimizer: optimizer.
        warmup_steps: number of warmup steps.
    """

    optimizer: optim.Optimizer
    warmup_steps: int

    def __init__(
        self,
        optimizer,
        warmup_steps,
    ):
        self.warmup_steps = warmup_steps
        self.decay_factor = numpy.sqrt(warmup_steps)
        super().__init__(optimizer, self.lr_lambda)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.optimizer}, {self.warmup_steps})"
        )

    def lr_lambda(self, step: int) -> float:
        """Computes the learning rate lambda at a given step.

         Args:
            step: current step.

        Returns:
            float: lr_lambda.
        """
        if self.warmup_steps < 1:
            return self.decay_factor
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5
