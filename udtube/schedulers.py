"""Custom schedulers."""

import numpy

from torch import optim

from . import defaults


class WarmupInverseSquareRootSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup steps, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    warmup_steps: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: defaults.WARMUP_STEPS,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            warmup_steps (int): number of warmup steps.
            **kwargs: ignored.
        """
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
            step (int): current step.

        Returns:
            float: lr_lambda.
        """
        if self.warmup_steps < 1:
            return self.decay_factor
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


class ReduceOnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce on plateau scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        reduceonplateau_factor: defaults.REDUCEONPLATEAU_FACTOR,
        reduceonplateau_patience: defaults.REDUCEONPLATEAU_PATIENCE,
        min_learning_rate: defaults.MIN_LEARNING_RATE,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            reduceonplateau_factor (float): factor by which the
                learning rate will be reduced: `new_lr *= factor`.
            reduceonplateau_patience (int): number of epochs with no
                improvement before reducing LR.
            min_learning_rate (float): lower bound on the learning rate.
            **kwargs: ignored.
        """
        super().__init__(
            optimizer,
            mode="min",
            factor=reduceonplateau_factor,
            patience=reduceonplateau_patience,
            min_lr=min_learning_rate,
        )
        self.metric = "val_loss"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.optimizer}, "
            f"{self.factor}, {self.patience}, {self.min_learning_rate})"
        )
