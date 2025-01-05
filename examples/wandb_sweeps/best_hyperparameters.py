#!/usr/bin/env python
"""Retrieve hyperparameters from a W&B sweep."""

import argparse
import logging
from typing import Any, Dict
import yaml

import wandb
import sweep

# Top-level categories allowed in Lightning CLI configs
ALLOWED_TOP_LEVEL_CATEGORIES = frozenset(
    ["data", "model", "trainer", "checkpoint", "prediction"]
)


def dot_to_nested_dict(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dot notation dictionary to nested dictionary and handle configs.

    Uses recursive insertion for clean nested dictionary creation while
    maintaining proper structure for optimizer/scheduler configs and enforcing
    allowed categories.
    """
    nested = {}

    for key, value in flat_dict.items():
        # Handle model-related top-level keys
        if key in {"dropout", "encoder", "pooling_layers", "use_xpos"}:
            sweep._recursive_insert(nested, f"model.{key}", value)
            continue

        # Handle optimizer/scheduler configs
        if key.startswith(("classifier_", "encoder_")):
            prefix, remainder = key.split("_", 1)
            model_key = f"model.{prefix}_{remainder}"
            if isinstance(value, dict):
                if "class_path" in value:
                    sweep._recursive_insert(nested, model_key, value)
                else:
                    current = nested.setdefault("model", {})
                    current = current.setdefault(
                        f"{prefix}_{remainder}",
                        {},
                    )
                    current.update(value)
            continue

        # Handle model-prefixed or allowed category keys
        if (
            key.startswith("model.")
            or any(
                key.startswith(f"{cat}.")
                for cat in ALLOWED_TOP_LEVEL_CATEGORIES
            )
            or key in ALLOWED_TOP_LEVEL_CATEGORIES
        ):
            sweep._recursive_insert(nested, key, value)

    return nested


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    sweep = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")
    best_run = sweep.best_run()
    logging.info("Best run URL: %s", best_run.url)

    # Get all config values from best run
    flat_config = {
        key: value
        for key, value in sorted(best_run.config.items())
        if value is not None and "/" not in key
    }

    # Convert to nested dictionary structure
    config = dot_to_nested_dict(flat_config)
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
