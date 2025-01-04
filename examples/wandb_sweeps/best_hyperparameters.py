#!/usr/bin/env python
"""Retrieve hyperparameters from a W&B sweep."""

import argparse
import logging
import yaml

import wandb

from udtube import defaults

# Expand as needed.
FLAGS_TO_IGNORE = frozenset(
    ["eval_metrics", "local_run_dir", "n_model_params"]
)


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    sweep = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")
    best_run = sweep.best_run()
    logging.info("Best run URL: %s", best_run.url)
    # Sorting for stability.
    config = {}
    for key, value in sorted(best_run.config.items()):
        # Exclusions:
        #
        # * Explicitly ignored flags
        # * Keys with "/" is for redundant parameters in the scheduler.
        # * Key/value pairs that are defaults can be omitted.
        # * Keys ending in "_cls", "_idx", and "vocab_size" are set
        #   automatically.
        # * None values are defaults definitionally.
        if key in FLAGS_TO_IGNORE:
            continue
        if "/" in key:
            continue
        key_upper = key.upper()
        if hasattr(defaults, key_upper):
            if getattr(defaults, key_upper) == value:
                continue
        excluded_suffixes = ["_cls", "_idx", "vocab_size"]
        if any(key.endswith(suffix) for suffix in excluded_suffixes):
            continue
        if value is None:
            continue
        config[key] = value
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        required=True,
        help="The entity scope for the project.",
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    main(parser.parse_args())
