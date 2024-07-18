import logging
import os
import subprocess
import sys

import yaml
import wandb
import functools
import argparse

os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep_id",
        required=True,
        help="ID for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument(
        "--count",
        type=int,
        help="The max number of runs for this agent",
    )
    parser.add_argument(
        "--fit_config",
        help="directory of the train config. Default is ../configs/fit_config.yaml",
        default="../configs/fit_config.yaml"
    )
    return parser

def setup_run_params(train_config='../configs/fit_config.yaml'):
    with open(train_config, 'r') as f:
        wandb.init()
        orig_config = yaml.safe_load(f)
        # Model arguments come from the wandb sweep config and override any
        # conflicting arguments passed via the CLI.
        for key, value in dict(wandb.config).items():
            sub_key = None
            if key.startswith("model."):
                sub_key = "model"
                key = key[len("model."):]
            elif key.startswith("trainer."):
                sub_key = "trainer"
                key = key[len("trainer."):]
            if key in orig_config or sub_key and key in orig_config[sub_key]:
                logging.info(f"Overriding YAML argument: {key}")
                if sub_key:
                    orig_config[sub_key][key] = value
                else:
                    orig_config[key] = value
        with open('temp.yaml', 'w') as file:
            yaml.dump(orig_config, file)
        logging.info("Running!")
        os.system("python3 udtube_package.py fit --config temp.yaml")


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    # Forces log_wandb to True, so that the PTL trainer logs runtime metrics to wandb.
    args.log_wandb = True
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            project=args.project,
            entity=args.entity,
            function=functools.partial(setup_run_params, train_config=args.fit_config),
        )
    except Exception as e:
        logging.error(e, file=sys.stderr)
        # Exits gracefully, so wandb logs the error.
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
