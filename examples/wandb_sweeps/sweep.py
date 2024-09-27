#!/usr/bin/env python
"""Runs a W&B sweep."""

import argparse
import logging
import traceback
import warnings

import torch
import wandb

from udtube import cli

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(args: argparse.Namespace) -> None:
    """Runs a single training run.

    The wandb config data used here comes from the environment.

    Args:
        args (argparse.Namespace).
    """
    wandb.init()
    arglist = [
        f"--{key} {value!r}" for key, value in dict(wandb.config).items()
    ]
    try:
        cli.fit_for_sweep(arglist)
    except RuntimeError as error:
        # TODO: consider specializing this further if a solution to
        # https://github.com/pytorch/pytorch/issues/48365 is accepted.
        logging.error("Runtime error: %s", error)
    finally:
        # Clears the CUDA cache.
        torch.cuda.empty_cache()


def main(args: argparse.Namespace) -> None:
    # TODO: manually enable W&B logger?
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=train_sweep,
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so W&B logs the error.
        logging.fatal(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    cli.config_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument("--count", type=int, help="Number of runs to perform.")
    main(parser.parser_args())
