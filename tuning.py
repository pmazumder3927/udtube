import os
import subprocess

import yaml
import wandb
import functools
import argparse

os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

def get_args():
    with open('configs/fit_config.yaml', 'r') as f:
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
                print(f"Overriding YAML argument: {key}")
                if sub_key:
                    orig_config[sub_key][key] = value
                else:
                    orig_config[key] = value
        with open('temp.yaml', 'w') as file:
            yaml.dump(orig_config, file)
        print("Running!")
        os.system("python3 udtube.py fit --config temp.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep_id",
        required=True,
        help="ID for the sweep to run the agent in.",
    )
    # parser.add_argument(
    #     "--entity", required=True, help="The entity scope for the project."
    # )
    # parser.add_argument(
    #     "--project", required=True, help="The project of the sweep."
    # )
    # parser.add_argument(
    #     "--count",
    #     type=int,
    #     help="The max number of runs for this agent",
    # )
    args = parser.parse_args()
    # Forces log_wandb to True, so that the PTL trainer logs runtime metrics
    # to wandb.
    args.log_wandb = True
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            project="HyperParamandTheSweepyHollow",
            function=functools.partial(get_args),
        )
    except Exception as e:
        print(e)
        print("pain")
        # Exits gracefully, so wandb logs the error.
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()